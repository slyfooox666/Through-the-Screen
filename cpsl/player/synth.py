"""Geometry-aware CPSL synthesis helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


def srgb_to_linear(value: np.ndarray) -> np.ndarray:
    """Convert sRGB values in [0,1] to linear space."""
    threshold = 0.04045
    below = value <= threshold
    out = np.empty_like(value, dtype=np.float32)
    out[below] = value[below] / 12.92
    out[~below] = ((value[~below] + 0.055) / 1.055) ** 2.4
    return out


def linear_to_srgb(value: np.ndarray) -> np.ndarray:
    """Convert linear values in [0,1] back to sRGB."""
    threshold = 0.0031308
    below = value <= threshold
    out = np.empty_like(value, dtype=np.float32)
    out[below] = value[below] * 12.92
    out[~below] = 1.055 * np.power(value[~below], 1 / 2.4) - 0.055
    return np.clip(out, 0.0, 1.0)


def build_intrinsic_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """Construct a 3x3 camera intrinsic matrix."""
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def plane_homography(
    Ks: np.ndarray,
    Kt: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    normal: np.ndarray,
    depth: float,
) -> np.ndarray:
    """Compute plane-induced homography H = K_t (R - t n^T / z) K_s^{-1}."""
    Ks_inv = np.linalg.inv(Ks)
    normal = normal.reshape(3, 1).astype(np.float32)
    t = t.reshape(3, 1).astype(np.float32)
    depth = float(depth if depth != 0 else 1e-3)
    middle = R.astype(np.float32) - (t @ normal.T) / depth
    H = Kt @ middle @ Ks_inv
    return H.astype(np.float32)


@dataclass
class WarpedLayer:
    color: np.ndarray  # premultiplied linear RGB
    alpha: np.ndarray  # scalar alpha
    depth: np.ndarray  # warped depth map


def warp_layer(
    color_srgb: np.ndarray,
    alpha: np.ndarray,
    depth_map: np.ndarray,
    H: np.ndarray,
    output_size: Tuple[int, int],
    edge_smooth_px: int = 2,
) -> WarpedLayer:
    """Warp layer colour/alpha/depth via the provided homography."""
    width, height = output_size
    # Convert premultiplied sRGB colour to linear space.
    color_linear = srgb_to_linear(np.clip(color_srgb, 0.0, 1.0)).astype(np.float32)
    warped_color = cv2.warpPerspective(
        color_linear,
        H,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    warped_alpha = cv2.warpPerspective(
        alpha.astype(np.float32),
        H,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    warped_depth = cv2.warpPerspective(
        depth_map.astype(np.float32),
        H,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=1.0,
    )

    if edge_smooth_px > 0:
        kernel = edge_smooth_px * 2 + 1
        original_alpha = warped_alpha.copy()
        blurred = cv2.GaussianBlur(warped_alpha, (kernel, kernel), 0.0)
        warped_alpha = np.maximum(warped_alpha, blurred)
        # Preserve premultiplied relationship.
        safe_original = np.where(original_alpha > 1e-6, original_alpha, 1.0)
        gain = np.where(warped_alpha > 1e-6, warped_alpha / safe_original, 0.0)
        warped_color *= gain[..., None]

    # Mask depth where alpha is low to avoid leaking background.
    warped_depth = np.where(warped_alpha > 1e-5, warped_depth, np.inf)

    return WarpedLayer(color=warped_color, alpha=warped_alpha, depth=warped_depth)


def composite_front_to_back(
    warped_layers: Sequence[WarpedLayer],
    output_size: Tuple[int, int],
    use_zbuffer: bool = True,
) -> np.ndarray:
    """Composite premultiplied layers using front-to-back blending with optional Z-buffer."""
    width, height = output_size
    accum_color = np.zeros((height, width, 3), dtype=np.float32)
    accum_alpha = np.zeros((height, width), dtype=np.float32)
    zbuffer = np.full((height, width), np.inf, dtype=np.float32) if use_zbuffer else None

    for layer in warped_layers:
        alpha = np.clip(layer.alpha, 0.0, 1.0)
        color = np.clip(layer.color, 0.0, 1.0)
        depth = layer.depth
        if use_zbuffer:
            mask = (alpha > 1e-5) & (depth < (zbuffer - 1e-5))
            if not np.any(mask):
                continue
            mask_f = mask.astype(np.float32)
            alpha = alpha * mask_f
            color = color * mask_f[..., None]
            zbuffer = np.where(mask, depth, zbuffer)

        accum_color = color + accum_color * (1.0 - alpha[..., None])
        accum_alpha = alpha + accum_alpha * (1.0 - alpha)

    nonzero = accum_alpha > 1e-5
    if np.any(nonzero):
        accum_color[nonzero] /= accum_alpha[nonzero, None]
    accum_color = np.clip(accum_color, 0.0, 1.0)
    accum_color = linear_to_srgb(accum_color)
    accum_color = np.clip(accum_color, 0.0, 1.0)
    return accum_color


def render_frame(
    layers: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray, float, Optional[np.ndarray]]],
    Ks: np.ndarray,
    Kt: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    output_size: Tuple[int, int],
    edge_smooth_px: int = 2,
    use_zbuffer: bool = True,
) -> np.ndarray:
    """Render novel view given per-layer data and camera parameters."""
    warped_layers: List[WarpedLayer] = []
    for color, alpha, depth_map, plane_depth, normal in layers:
        plane_normal = normal if normal is not None else np.array([0.0, 0.0, 1.0], dtype=np.float32)
        H = plane_homography(Ks, Kt, R, t, plane_normal, plane_depth)
        warped = warp_layer(
            color_srgb=color,
            alpha=alpha,
            depth_map=depth_map,
            H=H,
            output_size=output_size,
            edge_smooth_px=edge_smooth_px,
        )
        warped_layers.append(warped)

    accum_color = composite_front_to_back(
        warped_layers=warped_layers, output_size=output_size, use_zbuffer=use_zbuffer
    )
    accum_color = np.clip(accum_color * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(accum_color, cv2.COLOR_RGB2BGR)
