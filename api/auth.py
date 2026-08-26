"""Who is asking. A signed header, and the policy layer behind it.

    X-CaseFile-Persona: v1.<base64url(claims)>.<base64url(hmac-sha256)>

THIS IS A PROTOTYPE STAND-IN FOR ENTERPRISE SSO, AND IT IS SHAPED LIKE
ONE ON PURPOSE. In a deployment the header is replaced by an OIDC access
token from the customer's identity provider, validated against its JWKS.
Everything downstream of `authenticate()` — the directory lookup, the
persona resolution, the row predicate, the column mask, the audit row —
is unchanged by that swap, because none of it trusts the header for
anything except the claim of identity. That is the whole design: the
transport proves WHO, and the policy layer decides WHAT, and the second
never asks the first for permission.

WHAT THE SIGNATURE BUYS, AND WHAT IT DOES NOT. It proves the header was
minted by something holding the signing secret, and that it has not
expired. It does NOT establish that the persona in the token is a persona
the bearer holds. `authenticate()` resolves the user against `dim_user`
and refuses a token whose persona disagrees with the directory — so a
correctly signed token claiming `cxo` for a store manager's user id is
rejected, not honoured. A signature that could mint authority would make
the directory decorative.

WHY IDENTITY RESOLUTION DOES NOT GO THROUGH THE GOVERNED DOOR.
`execute_governed` resolves a persona's row predicate before it reads
anything. It cannot be used to find out WHICH persona is asking, because
it needs the answer to that question in order to run. So the directory
read here is a plain statement against `dim_user`, which holds no
measure and no business fact — only who exists and what they are. The
first governed read of the request happens immediately after, with the
resolved user, exactly as rule 5 requires.

THE SIGNING SECRET. `CASEFILE_SIGNING_SECRET` from the environment. With
none set, a random one is generated per process: tokens then work for the
life of that process and stop working when it restarts, which is correct
for a prototype and makes it impossible to ship a default secret by
accident. There is deliberately no fallback constant to forget to change.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Mapping

import duckdb

from security.policy import User

#: The header. Named for what it carries rather than for `Authorization`,
#: so that adding a real bearer token later does not collide with it.
HEADER = "X-CaseFile-Persona"

#: Token format version, so a change to the claim set is detectable rather
#: than a confusing signature failure.
VERSION = "v1"

SEPARATOR = "."

#: Environment variable holding the shared secret.
SECRET_ENV = "CASEFILE_SIGNING_SECRET"

#: How long a minted token stays valid. Short because the prototype mints
#: them on demand; an SSO deployment takes the lifetime from the IdP's
#: token instead and this constant stops being used.
DEFAULT_TTL = timedelta(hours=12)

#: Bytes of entropy in a generated per-process secret.
_GENERATED_SECRET_BYTES = 32

#: The directory. Identity, not business data — see the module docstring
#: on why this read does not go through `execute_governed`.
DIRECTORY_TABLE = "dim_user"
DIRECTORY_QUERY = (
    "SELECT user_id, display_name, email, persona, region, store_id "
    f"FROM {DIRECTORY_TABLE} WHERE user_id = ?"
)


class AuthError(PermissionError):
    """The caller did not establish who they are."""


# ---------------------------------------------------------------------------
# The secret
# ---------------------------------------------------------------------------

_PROCESS_SECRET: str | None = None


def signing_secret(env: Mapping[str, str] | None = None) -> str:
    """The signing secret, from the environment or generated once.

    A generated secret is not a fallback that quietly works in
    production — it is a secret nobody else knows, so every token minted
    against it dies with the process. `signing_secret_is_ephemeral()`
    reports which of the two is in use, and the app logs it at startup.
    """
    global _PROCESS_SECRET
    source = os.environ if env is None else env
    configured = source.get(SECRET_ENV, "").strip()
    if configured:
        return configured
    if _PROCESS_SECRET is None:
        _PROCESS_SECRET = secrets.token_urlsafe(_GENERATED_SECRET_BYTES)
    return _PROCESS_SECRET


def signing_secret_is_ephemeral(env: Mapping[str, str] | None = None) -> bool:
    source = os.environ if env is None else env
    return not source.get(SECRET_ENV, "").strip()


def reset_secret() -> None:
    """Forget a generated secret. For tests."""
    global _PROCESS_SECRET
    _PROCESS_SECRET = None


# ---------------------------------------------------------------------------
# Claims
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Claims:
    """What the token asserts. Asserted, not proven — see `authenticate`."""

    user_id: str
    persona: str
    issued_at: datetime
    expires_at: datetime

    def expired(self, now: datetime | None = None) -> bool:
        return (now or datetime.now(UTC)) >= self.expires_at

    def as_payload(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "persona": self.persona,
            "issued_at": self.issued_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Claims:
        try:
            return cls(
                user_id=str(payload["user_id"]),
                persona=str(payload["persona"]),
                issued_at=datetime.fromisoformat(str(payload["issued_at"])),
                expires_at=datetime.fromisoformat(str(payload["expires_at"])),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise AuthError(f"token claims are malformed: {exc}") from exc


def mint(
    user_id: str,
    persona: str,
    *,
    ttl: timedelta = DEFAULT_TTL,
    now: datetime | None = None,
    secret: str | None = None,
) -> str:
    """Sign a token for `user_id`. The prototype's stand-in for an IdP.

    Deliberately exported: the demo, the tests and the frontend's dev
    login all need to produce one, and a minting path that only exists
    inside the tests is a minting path nobody has checked.
    """
    issued = now or datetime.now(UTC)
    claims = Claims(
        user_id=user_id, persona=persona, issued_at=issued, expires_at=issued + ttl
    )
    payload = _encode(json.dumps(claims.as_payload(), sort_keys=True).encode("utf-8"))
    signature = _sign(payload, secret or signing_secret())
    return SEPARATOR.join((VERSION, payload, signature))


def verify(token: str, *, secret: str | None = None, now: datetime | None = None) -> Claims:
    """Check the signature and the expiry. Returns the claims, unproven.

    Raises `AuthError` for every failure mode with the same shape of
    message: the caller learns that the token was not accepted, not which
    of the checks it failed. Distinguishing "bad signature" from "unknown
    user" for an unauthenticated caller is a free oracle.
    """
    parts = token.strip().split(SEPARATOR)
    if len(parts) != 3:
        raise AuthError(f"{HEADER} is not a {VERSION} token")
    version, payload, signature = parts
    if version != VERSION:
        raise AuthError(f"{HEADER} is not a {VERSION} token")

    expected = _sign(payload, secret or signing_secret())
    # Constant time: a comparison that returns early leaks the signature
    # one byte at a time to anyone willing to measure.
    if not hmac.compare_digest(signature, expected):
        raise AuthError(f"{HEADER} signature does not verify")

    try:
        decoded = json.loads(_decode(payload))
    except (ValueError, UnicodeDecodeError) as exc:
        raise AuthError(f"{HEADER} payload is not readable: {exc}") from exc

    claims = Claims.from_payload(decoded)
    if claims.expired(now):
        raise AuthError(f"{HEADER} expired at {claims.expires_at.isoformat()}")
    return claims


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


def authenticate(
    connection: duckdb.DuckDBPyConnection,
    token: str,
    *,
    secret: str | None = None,
    now: datetime | None = None,
) -> User:
    """Verify the token, then resolve it against the directory.

    TWO INDEPENDENT CHECKS, AND BOTH MUST HOLD. The signature says the
    token was minted by us. The directory says the user exists and holds
    the persona claimed. A token that passes the first and fails the
    second is refused — that is the case where somebody with a valid
    token tries to be somebody else, and it is the only interesting
    attack against a scheme like this one.
    """
    claims = verify(token, secret=secret, now=now)

    row = connection.execute(DIRECTORY_QUERY, [claims.user_id]).fetchone()
    if row is None:
        raise AuthError(
            f"{claims.user_id} is not in {DIRECTORY_TABLE}. A signed token is not "
            "an account."
        )
    columns = ("user_id", "display_name", "email", "persona", "region", "store_id")
    record = dict(zip(columns, row, strict=True))

    if str(record["persona"]) != claims.persona:
        raise AuthError(
            f"the token claims persona {claims.persona!r} for {claims.user_id}, and the "
            f"directory says {record['persona']!r}. The directory decides."
        )
    return User.from_row(record)


def _sign(payload: str, secret: str) -> str:
    digest = hmac.new(secret.encode("utf-8"), payload.encode("ascii"), hashlib.sha256)
    return _encode(digest.digest())


def _encode(raw: bytes) -> str:
    """base64url without padding. Padding is not URL- or header-safe."""
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _decode(value: str) -> str:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding).decode("utf-8")


__all__ = [
    "DEFAULT_TTL",
    "DIRECTORY_TABLE",
    "HEADER",
    "SECRET_ENV",
    "VERSION",
    "AuthError",
    "Claims",
    "authenticate",
    "mint",
    "reset_secret",
    "signing_secret",
    "signing_secret_is_ephemeral",
    "verify",
]
