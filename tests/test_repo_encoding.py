"""Every tracked text file is real UTF-8.

This exists because README.md was UTF-16 twice. The first time it had a
BOM, which made hatchling fail loudly with a UnicodeDecodeError. The
second time the BOM was gone, and that is the dangerous case: UTF-16LE
bytes without a BOM decode as *valid* UTF-8 — every other byte is a NUL,
which is a legal code point — so nothing raised, the package built, and
the file simply rendered as `C a s e F i l e . a i` on GitHub.

A NUL byte in a text file is the tell. Nothing here should contain one.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

TEXT_SUFFIXES = {
    ".py", ".md", ".yaml", ".yml", ".toml", ".json", ".txt", ".cfg", ".ini",
    ".gitignore", ".gitattributes",
}
TEXT_NAMES = {"Makefile", ".gitignore", ".gitattributes"}


def tracked_text_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    )
    files = []
    for name in result.stdout.decode("utf-8").split("\0"):
        if not name:
            continue
        path = ROOT / name
        if not path.is_file():
            continue
        if path.suffix in TEXT_SUFFIXES or path.name in TEXT_NAMES:
            files.append(path)
    return sorted(files)


def test_there_are_files_to_check():
    """Guard against the scan passing because git returned nothing."""
    assert len(tracked_text_files()) > 20


@pytest.mark.parametrize(
    "path", tracked_text_files(), ids=lambda p: p.relative_to(ROOT).as_posix()
)
def test_file_is_utf8_without_nul_bytes(path: Path):
    raw = path.read_bytes()

    assert b"\x00" not in raw, (
        f"{path.relative_to(ROOT)} contains NUL bytes — it is almost certainly "
        "UTF-16. Convert it with:\n"
        "  python -c \"import pathlib;p=pathlib.Path(FILE);"
        "p.write_bytes(p.read_bytes().decode('utf-16').encode('utf-8'))\"\n"
        "Note that some editors PRESERVE a file's existing encoding on save, "
        "so rewriting the content is not enough — write the bytes."
    )

    try:
        raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        pytest.fail(f"{path.relative_to(ROOT)} is not valid UTF-8: {exc}")


def test_readme_is_readable_utf8():
    """The specific file that broke the build backend, pinned by name."""
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert text.startswith("# CaseFile.ai")
    assert "\x00" not in text


def test_claude_md_is_readable_utf8():
    """The source of truth must be readable, or nothing else matters."""
    text = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    assert text.startswith("# CLAUDE.md")
    assert "The ten non-negotiable rules" in text
