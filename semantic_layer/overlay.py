"""The runtime overlay — generated semantic-layer files.

`semantic_layer/*.yaml` is written by people and read by the engine.
`semantic_layer/runtime/*.yaml` is written by the engine and read by
people. Both are semantic-layer files, so both live here and the
serialisation of both is this package's business.

WHY THIS IS NOT IN `engine/learn/`. Rule 2: engine modules load
thresholds through `semantic_layer.schema`, never by opening YAML
themselves, and `tests/test_no_thresholds_in_engine.py` asserts that no
module under `engine/` imports `yaml` at all. Writing a generated file is
not the abuse that rule guards against — but the rule is better kept
simple and absolute than qualified with an exception for one module, and
the semantic layer is the honest owner of its own directory anyway.

WHY GENERATE A FILE AT ALL. Because "the system has quietly decided
stock-outs are less likely than we wrote down" is exactly the kind of
change that should show up in a diff and be reviewable. A learned value
that lives only in a database column is a change to the engine's
judgement that no code review will ever see.

Nothing here is authoritative. Every file this module writes can be
deleted and regenerated from the warehouse. If the two ever disagree, the
warehouse is right and this is stale.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import yaml

from semantic_layer.schema import PACKAGE_ROOT

#: Generated files live apart from the hand-written contracts, so nobody
#: has to remember which of the two a filename is.
OVERLAY_DIR = PACKAGE_ROOT / "runtime"
OVERLAY_PATH = OVERLAY_DIR / "priors.yaml"

#: Overlay format version. Bumped when the shape changes, so a reader can
#: tell a stale file from a current one rather than mis-parsing it.
OVERLAY_VERSION = 1

PRIORS_HEADER = """\
# GENERATED — do not edit. Rewritten whenever driver feedback lands.
#
# The runtime overlay over `causal_graph.yaml`'s declared priors. Every
# figure here is derived from two things that ARE editable: the declared
# prior in the causal graph, and the counts in the `hypothesis_prior`
# table. Delete this file and it regenerates; edit it and the next
# feedback event overwrites your edit.
#
# It exists so that a prior the system has moved shows up in a diff. A
# learned value that lives only in a database column is a change to the
# engine's judgement that no code review will ever see.
#
# `effective` is the Beta posterior mean of the declared prior against the
# observations, bounded by `learning.yaml -> priors`. `shift` is how far
# feedback moved it. `capped: true` means the bound is binding and further
# feedback will not move it — at which point the causal graph itself needs
# a person to look at it, which is the correct escalation.
"""

#: Decimal places in a written probability. Display precision for a file a
#: human reads and diffs — the values themselves are recomputed from the
#: counts on every read, never from this file.
PLACES = 6


def write_priors_overlay(
    entries: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    path: Path | None = None,
    generated_at: datetime,
    rule: Mapping[str, Any],
) -> Path | None:
    """Materialise the priors overlay. `None` when it could not be written.

    BEST EFFORT, DELIBERATELY. The warehouse holds the counts; this file
    is a view of them. A read-only filesystem, a container without a
    writable mount, a permissions problem — none of those should fail a
    reader's feedback, because the observation has already been recorded
    and the overlay regenerates on the next successful write.
    """
    target = path or OVERLAY_PATH
    payload = {
        "version": OVERLAY_VERSION,
        "generated_at": generated_at.isoformat(),
        "source": "hypothesis_prior",
        "rule": dict(rule),
        "priors": {
            kpi: {
                hypothesis: _rounded(values)
                for hypothesis, values in sorted(by_hypothesis.items())
            }
            for kpi, by_hypothesis in sorted(entries.items())
        },
    }
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            PRIORS_HEADER
            + yaml.safe_dump(payload, sort_keys=True, default_flow_style=False),
            encoding="utf-8",
        )
    except OSError:
        return None
    return target


def read_priors_overlay(path: Path | None = None) -> dict:
    """The overlay as it stands on disk. `{}` when it has never been written."""
    target = path or OVERLAY_PATH
    if not target.exists():
        return {}
    loaded = yaml.safe_load(target.read_text(encoding="utf-8"))
    return loaded or {}


def _rounded(values: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: round(value, PLACES) if isinstance(value, float) else value
        for key, value in values.items()
    }


__all__ = [
    "OVERLAY_DIR",
    "OVERLAY_PATH",
    "OVERLAY_VERSION",
    "PLACES",
    "PRIORS_HEADER",
    "read_priors_overlay",
    "write_priors_overlay",
]
