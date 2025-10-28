"""Lightweight morphology helpers used by Dynamic Pixel Strip."""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import cupy as cp
    from cupyx.scipy.ndimage import maximum_filter as cp_max_filter
    from cupyx.scipy.ndimage import minimum_filter as cp_min_filter
except Exception:  # pragma: no cover - optional dependency
    cp = None  # type: ignore[assignment]
    cp_max_filter = None  # type: ignore[assignment]
    cp_min_filter = None  # type: ignore[assignment]

import cv2

_EPS = 1e-6


def _select_backend(backend: str, array: np.ndarray):
    if backend == "cupy" or (backend == "auto" and cp is not None and isinstance(array, cp.ndarray)):
        if cp is None:
            raise RuntimeError("CuPy requested but not installed.")
        return cp
    return np


def dilate(mask: np.ndarray, radius: int, backend: str = "auto") -> np.ndarray:
    """Binary dilation with an elliptical kernel."""
    if radius <= 0:
        return mask.copy()
    xp = _select_backend(backend, mask)
    if xp is np:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
        return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    if cp_max_filter is None:
        raise RuntimeError("CuPy maximum filter unavailable.")
    size = radius * 2 + 1
    return cp_max_filter(mask.astype(xp.uint8), size=size, mode="nearest") > 0


def erode(mask: np.ndarray, radius: int, backend: str = "auto") -> np.ndarray:
    """Binary erosion with an elliptical kernel."""
    if radius <= 0:
        return mask.copy()
    xp = _select_backend(backend, mask)
    if xp is np:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
        return cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    if cp_min_filter is None:
        raise RuntimeError("CuPy minimum filter unavailable.")
    size = radius * 2 + 1
    return cp_min_filter(mask.astype(xp.uint8), size=size, mode="nearest") > 0


def feather_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """
    Generate a 0..1 feather weight inside the mask.

    Weight is 0 at the boundary and ramps to 1 over ``radius`` pixels.
    """
    if radius <= 0:
        return mask.astype(np.float32)
    mask_uint8 = mask.astype(np.uint8)
    dist = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 3)
    norm = radius if radius > 0 else 1
    return np.clip(dist / (norm + _EPS), 0.0, 1.0).astype(np.float32)

