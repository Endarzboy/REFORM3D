"""Atomic JSON/CSV writers for readout-sensitivity artifacts.

Partial artifacts are never left behind: each file is written to a sibling
temporary path and atomically renamed into place.
"""

from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def write_json(path: Path, payload: Any) -> None:
    """Atomically write ``payload`` as sorted, indented JSON."""
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    _atomic_write_bytes(Path(path), text.encode("utf-8"))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    """Atomically write ``rows`` as CSV with the given ``fieldnames``."""
    import io

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(fieldnames))
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key) for key in fieldnames})
    _atomic_write_bytes(Path(path), buffer.getvalue().encode("utf-8"))


def write_text(path: Path, text: str) -> None:
    """Atomically write UTF-8 text (used for Markdown tables)."""
    _atomic_write_bytes(Path(path), text.encode("utf-8"))


def read_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)
