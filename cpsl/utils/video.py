"""Video IO utilities built on OpenCV."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterable, Optional, Tuple

import cv2
import numpy as np


ColourFrame = np.ndarray


@dataclass
class VideoReader:
    """Simple wrapper around OpenCV's video capture API."""

    path: Path
    stride: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.path, str):
            self.path = Path(self.path)
        if not self.path.exists():
            raise FileNotFoundError(f"Video not found: {self.path}")
        self._cap: Optional[cv2.VideoCapture] = None

    def __enter__(self) -> "VideoReader":
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.path}")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def frames(self) -> Generator[Tuple[int, ColourFrame], None, None]:
        """Iterate over (index, frame) pairs in BGR format."""
        if self._cap is None:
            raise RuntimeError("VideoReader must be used as a context manager")
        idx = 0
        frame_idx = 0
        while True:
            ret, frame = self._cap.read()
            if not ret:
                break
            if frame_idx % self.stride == 0:
                yield idx, frame
                idx += 1
            frame_idx += 1

    @property
    def metadata(self) -> Tuple[int, int, float]:
        """Return (width, height, fps) for the video."""
        if self._cap is None:
            raise RuntimeError("VideoReader must be used as a context manager")
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        return width, height, fps


@dataclass
class VideoWriter:
    """Write a sequence of frames to a video file."""

    path: Path
    fps: float
    frame_size: Tuple[int, int]
    codec: str = "mp4v"

    def __post_init__(self) -> None:
        if isinstance(self.path, str):
            self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self._writer = cv2.VideoWriter(str(self.path), fourcc, self.fps, self.frame_size)
        if not self._writer.isOpened():
            raise RuntimeError(f"Failed to open writer for {self.path}")

    def write_frames(self, frames: Iterable[ColourFrame]) -> None:
        for frame in frames:
            if frame.shape[1::-1] != self.frame_size:
                raise ValueError(
                    f"Frame size {frame.shape[1::-1]} does not match expected {self.frame_size}"
                )
            self._writer.write(frame)

    def release(self) -> None:
        self._writer.release()

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


