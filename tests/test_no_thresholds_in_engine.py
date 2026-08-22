"""Rule 2, enforced: no threshold lives in Python.

CLAUDE.md says "a `grep` for hardcoded thresholds in `engine/` must return
nothing". A literal text grep cannot tell a business threshold from a
schema bound — `Field(ge=0.0, le=1.0)` is a type constraint, not a
materiality limit — so this walks the AST instead and reports `file:line`
for anything that is genuinely a magic number.

What is allowed, and why:

  0 and 1 (and 0.0 / 1.0)   identity and zero elements, empty checks,
                            probability endpoints. Not tunable.
  Field(ge=…, le=…, gt=…,   Pydantic schema bounds. These constrain the
  lt=…, min_length=…, …)    TYPE, not the business rule. `p_value` being
                            in [0, 1] is arithmetic, not policy.

Everything else fails, and the fix is always the same: move the number to
`semantic_layer/*.yaml` and read it from the loaded layer.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ENGINE_DIR = Path(__file__).resolve().parents[1] / "engine"

#: Numbers that carry no policy. Kept deliberately tiny.
STRUCTURAL_VALUES = {0, 1, 0.0, 1.0}

#: Pydantic keyword arguments whose values are schema bounds, not thresholds.
SCHEMA_BOUND_KWARGS = {
    "ge",
    "le",
    "gt",
    "lt",
    "min_length",
    "max_length",
    "multiple_of",
    "max_digits",
    "decimal_places",
    "min_items",
    "max_items",
}


def _is_field_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Name):
        return func.id == "Field"
    if isinstance(func, ast.Attribute):
        return func.attr == "Field"
    return False


def _schema_bound_nodes(tree: ast.AST) -> set[int]:
    """ids of constant nodes that are Pydantic schema bounds."""
    allowed: set[int] = set()
    for node in ast.walk(tree):
        if not _is_field_call(node):
            continue
        for keyword in node.keywords:
            if keyword.arg in SCHEMA_BOUND_KWARGS:
                for sub in ast.walk(keyword.value):
                    allowed.add(id(sub))
    return allowed


def _magic_numbers(path: Path) -> list[tuple[int, object]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    allowed = _schema_bound_nodes(tree)
    findings: list[tuple[int, object]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant):
            continue
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            continue
        if node.value in STRUCTURAL_VALUES:
            continue
        if id(node) in allowed:
            continue
        findings.append((node.lineno, node.value))
    return findings


def engine_sources() -> list[Path]:
    return sorted(p for p in ENGINE_DIR.rglob("*.py") if "__pycache__" not in p.parts)


def test_engine_has_sources_to_scan():
    """Guard against the scan silently passing because it found no files."""
    assert engine_sources(), f"no Python sources found under {ENGINE_DIR}"


@pytest.mark.parametrize("path", engine_sources(), ids=lambda p: p.name)
def test_no_hardcoded_thresholds(path: Path):
    findings = _magic_numbers(path)
    if findings:
        listed = "\n".join(
            f"  {path.relative_to(ENGINE_DIR.parent)}:{line}  ->  {value!r}"
            for line, value in findings
        )
        pytest.fail(
            "CLAUDE.md rule 2: no threshold may live in Python.\n"
            f"Magic numbers found in {path.name}:\n{listed}\n"
            "Move each to semantic_layer/*.yaml and read it from the loaded layer."
        )


def test_engine_does_not_import_yaml_directly():
    """Thresholds arrive through the schema loader, not ad-hoc YAML reads.

    An engine module that opens a YAML file itself bypasses validation and
    can pick up a value the schema would have rejected.
    """
    offenders = []
    for path in engine_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(name.split(".")[0] == "yaml" for name in names):
                offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, (
        "engine modules must load thresholds via semantic_layer.schema, "
        f"not by reading YAML directly: {offenders}"
    )
