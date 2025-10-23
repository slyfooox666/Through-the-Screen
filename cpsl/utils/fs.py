"""Filesystem helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def ensure_dir(path: Path) -> None:
    """Create the directory if it does not yet exist."""
    path.mkdir(parents=True, exist_ok=True)


def save_json(data: Dict[str, Any], path: Path) -> None:
    """Serialize a dictionary to JSON with deterministic formatting."""
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)


