"""Image and color-space utilities for CPSL."""

from __future__ import annotations

from typing import Tuple

import numpy as np

_EPS = 1e-6


def to_linear_srgb(value: np.ndarray) -> np.ndarray:
    """Convert sRGB values in [0,1] to linear space."""
    if value.dtype != np.float32:
        value = value.astype(np.float32)
    threshold = 0.04045
    below = value <= threshold
    out = np.empty_like(value, dtype=np.float32)
    out[below] = value[below] / 12.92
    out[~below] = ((value[~below] + 0.055) / 1.055) ** 2.4
    return out


def to_srgb_linear(value: np.ndarray) -> np.ndarray:
    """Convert linear values in [0,1] back to sRGB."""
    if value.dtype != np.float32:
        value = value.astype(np.float32)
    threshold = 0.0031308
    below = value <= threshold
    out = np.empty_like(value, dtype=np.float32)
    out[below] = value[below] * 12.92
    out[~below] = 1.055 * np.power(value[~below], 1 / 2.4) - 0.055
    return np.clip(out, 0.0, 1.0)


def to_premul(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Premultiply RGB by alpha."""
    if rgb.dtype != np.float32:
        rgb = rgb.astype(np.float32)
    if alpha.dtype != np.float32:
        alpha = alpha.astype(np.float32)
    return np.dstack((rgb * alpha[..., None], alpha))


def from_premul(rgba: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Separate premultiplied RGBA into RGB and alpha."""
    if rgba.dtype != np.float32:
        rgba = rgba.astype(np.float32)
    alpha = rgba[..., 3]
    rgb = np.zeros_like(rgba[..., :3], dtype=np.float32)
    nonzero = alpha > _EPS
    rgb[nonzero] = rgba[..., :3][nonzero] / alpha[nonzero, None]
    return rgb, alpha


def composite_over(dst_rgba: np.ndarray, src_rgba: np.ndarray) -> np.ndarray:
    """
    Premultiplied over operation.

    out = src + dst * (1 - src.a)
    """
    if dst_rgba.dtype != np.float32:
        dst_rgba = dst_rgba.astype(np.float32)
    if src_rgba.dtype != np.float32:
        src_rgba = src_rgba.astype(np.float32)

    src_alpha = src_rgba[..., 3]
    inv_src_alpha = 1.0 - src_alpha

    out_rgb = src_rgba[..., :3] + dst_rgba[..., :3] * inv_src_alpha[..., None]
    out_alpha = src_alpha + dst_rgba[..., 3] * inv_src_alpha

    out = np.dstack((out_rgb, out_alpha)).astype(np.float32)
    out[..., :3] = np.clip(out[..., :3], 0.0, 1.0)
    out[..., 3] = np.clip(out[..., 3], 0.0, 1.0)
    return out


def ensure_rgba(array: np.ndarray) -> np.ndarray:
    """Ensure the array is HxWx4 float32 RGBA."""
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        raise ValueError("Expected array of shape HxWx3 or HxWx4.")
    if array.dtype != np.float32:
        array = array.astype(np.float32)
    if array.shape[2] == 3:
        alpha = np.ones(array.shape[:2], dtype=np.float32)
        array = np.dstack((array, alpha))
    return array

