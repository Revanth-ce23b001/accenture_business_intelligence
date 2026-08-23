"""The warehouse: schema, loaders and the reconciliation report.

`load.py` builds it from `data/raw`; `reconcile.py` measures the four
disagreements the source systems carry and writes them to
`data_gap_register` rather than repairing them.

Every connection comes from `engine/db.py` (CLAUDE.md rule 5).
"""

from engine.warehouse.load import LoadReport, WarehouseError, build_warehouse

__all__ = ["LoadReport", "WarehouseError", "build_warehouse"]
