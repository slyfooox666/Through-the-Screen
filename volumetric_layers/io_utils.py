"""Utility helpers for CPSL pipelines."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

_TIME_FACTOR = 1000.0


def parse_time_spec(spec: str) -> float:
    """Convert a human-readable time string into milliseconds.

    Accepted formats
    ----------------
    - ``"1234"`` or ``"1234ms"`` → 1234 milliseconds.
    - ``"HH:MM:SS.mmm"`` with optional hours and milliseconds components.

    Parameters
    ----------
    spec:
        Time specification.

    Returns
    -------
    float
        Milliseconds represented by ``spec``.

    Raises
    ------
    ValueError
        If the string cannot be parsed.
    """

    if spec is None:
        raise ValueError("Time spec cannot be None")

    value = spec.strip()
    if not value:
        raise ValueError("Time spec cannot be empty")

    if value.lower().endswith("ms"):
        value = value[:-2]

    # Plain integer / float in milliseconds
    try:
        if ":" not in value and value.replace(".", "", 1).replace("-", "", 1).isdigit():
            return float(value)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Invalid millisecond value: {spec}") from exc

    parts = value.split(":")
    if not 1 <= len(parts) <= 3:
        raise ValueError(f"Unsupported time format: {spec}")

    # Pad to [hours, minutes, seconds]
    parts = [float(p) if i == len(parts) - 1 else int(p) for i, p in enumerate(parts)]
    while len(parts) < 3:
        parts.insert(0, 0)

    hours, minutes, seconds = parts
    if minutes >= 60 or seconds < 0 or minutes < 0:
        raise ValueError(f"Invalid time spec: {spec}")

    total_ms = ((hours * 60 + minutes) * 60 + seconds) * _TIME_FACTOR
    return float(total_ms)


def ensure_dir(path: str | Path) -> Path:
    """Create ``path`` if missing and return it as :class:`Path`."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_json(path: str | Path, obj: Any) -> None:
    """Persist ``obj`` to ``path`` using a safe atomic write."""

    target = Path(path)
    ensure_dir(target.parent)
    tmp_path = target.with_suffix(target.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2)
    os.replace(tmp_path, target)
