"""JSONL index management for CPSL frame archives."""
from __future__ import annotations

import json
from bisect import bisect_left
from pathlib import Path
from typing import Dict, List, Optional

from .io_utils import ensure_dir


class CPSLIndex:
    """Append-only JSONL index tracking CPSL frames."""

    def __init__(self, root: str | Path, index_path: str | Path | None = None, mode: str = "a") -> None:
        self.root = Path(root)
        ensure_dir(self.root)
        self.path = Path(index_path) if index_path is not None else self.root / "index.jsonl"
        ensure_dir(self.path.parent)

        self._records: List[Dict[str, float | int | str]] = []
        if self.path.exists() and mode in {"a", "r"}:
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    payload = json.loads(line)
                    self._records.append(payload)
            # Ensure records ordered by pts_ms
            self._records.sort(key=lambda rec: rec["pts_ms"])

        if mode == "w":
            self.path.unlink(missing_ok=True)
            self._records.clear()

        file_mode = "a" if mode in {"a", "w"} else "r"
        self._handle = self.path.open(file_mode, encoding="utf-8") if file_mode != "r" else None
        self._last_pts = self._records[-1]["pts_ms"] if self._records else None

    def close(self) -> None:
        """Flush and close the underlying file handle."""

        if self._handle is not None and not self._handle.closed:
            self._handle.flush()
            self._handle.close()

    def __del__(self) -> None:  # pragma: no cover - destructor safety
        self.close()

    def __len__(self) -> int:
        return len(self._records)

    def add(self, frame_index: int, pts_ms: float, relpath: str, num_layers: int) -> Dict[str, float | int | str]:
        """Append a new record to the index and return it."""

        if self._handle is None:
            raise RuntimeError("Index opened in read-only mode")

        pts = float(pts_ms)
        if self._last_pts is not None and pts <= self._last_pts:
            pts = self._last_pts + 1e-3
        record = {
            "frame_index": int(frame_index),
            "pts_ms": pts,
            "relpath": relpath,
            "num_layers": int(num_layers),
        }
        self._records.append(record)
        self._last_pts = pts

        self._handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        self._handle.flush()
        return record

    def nearest(self, req_ms: float, how: str = "nearest") -> Dict[str, float | int | str]:
        """Return the closest record to ``req_ms``.

        Parameters
        ----------
        req_ms:
            Requested timestamp in milliseconds.
        how:
            One of ``"nearest"``, ``"floor"``, ``"ceil"``.
        """

        if not self._records:
            raise ValueError("Index is empty")

        how = how.lower()
        valid = {"nearest", "floor", "ceil"}
        if how not in valid:
            raise ValueError(f"Invalid seek mode '{how}'. Expected one of {sorted(valid)}")

        pts_list = [rec["pts_ms"] for rec in self._records]
        pos = bisect_left(pts_list, req_ms)

        if how == "floor":
            idx = max(0, pos - 1 if pos == len(pts_list) or pts_list[pos] > req_ms else pos)
        elif how == "ceil":
            idx = min(len(pts_list) - 1, pos if pos < len(pts_list) else len(pts_list) - 1)
        else:  # nearest
            if pos == 0:
                idx = 0
            elif pos >= len(pts_list):
                idx = len(pts_list) - 1
            else:
                before = pts_list[pos - 1]
                after = pts_list[pos]
                idx = pos if abs(after - req_ms) < abs(req_ms - before) else pos - 1

        return dict(self._records[idx])
