"""Geometry-aware CPSL synthesis helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from cpsl.player.dps import DPSParams, LayerView, apply_dynamic_strip, build_dynamic_strip


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
    """Warp layer colour/alpha/depth via the provided homography.

    Note: Input colour is stored in assets as premultiplied sRGB. We convert
    to unpremultiplied sRGB, then to linear, and re-premultiply after warping
    to maintain correct colour math.
    """
    width, height = output_size

    # Warp sRGB premultiplied colour and alpha separately.
    warped_color_srgb = cv2.warpPerspective(
        np.clip(color_srgb.astype(np.float32), 0.0, 1.0),
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

    # Edge smoothing on alpha.
    if edge_smooth_px > 0:
        kernel = edge_smooth_px * 2 + 1
        blurred = cv2.GaussianBlur(warped_alpha, (kernel, kernel), 0.0)
        warped_alpha = np.maximum(warped_alpha, blurred)

    # Convert to linear, re-premultiply by (possibly smoothed) alpha.
    safe_alpha = np.where(warped_alpha > 1e-6, warped_alpha, 1.0)
    unpremul_srgb = np.where(warped_alpha[..., None] > 1e-6, warped_color_srgb / safe_alpha[..., None], 0.0)
    unpremul_linear = srgb_to_linear(np.clip(unpremul_srgb, 0.0, 1.0)).astype(np.float32)
    warped_color_linear_premul = unpremul_linear * warped_alpha[..., None]

    # Mask depth where alpha is low to avoid leaking background.
    warped_depth = np.where(warped_alpha > 1e-5, warped_depth, np.inf)

    return WarpedLayer(color=warped_color_linear_premul, alpha=warped_alpha, depth=warped_depth)


def composite_front_to_back(
    warped_layers: Sequence[WarpedLayer],
    output_size: Tuple[int, int],
    use_zbuffer: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Composite premultiplied layers using correct front-to-back blending.

    We accumulate from front to back as:
        C_acc = C_acc + C_src * (1 - A_acc)
        A_acc = A_acc + A_src * (1 - A_acc)

    Optional Z-guard only prevents contributions where a strictly farther
    layer would overwrite already-opaque pixels; it never blocks backgrounds
    from filling holes.
    """
    width, height = output_size
    accum_color = np.zeros((height, width, 3), dtype=np.float32)
    accum_alpha = np.zeros((height, width), dtype=np.float32)
    zbuffer = np.full((height, width), np.inf, dtype=np.float32) if use_zbuffer else None

    for layer in warped_layers:
        alpha = np.clip(layer.alpha, 0.0, 1.0)
        color = np.clip(layer.color, 0.0, 1.0)

        if use_zbuffer:
            # Only allow contributions where the output isn't opaque yet.
            # Background is allowed to fill holes regardless of depth; we keep
            # a zbuffer only for diagnostics or future guards.
            not_opaque = accum_alpha < (1.0 - 1e-5)
            mask = (alpha > 1e-5) & not_opaque
            if np.any(mask):
                scale = np.zeros_like(alpha, dtype=np.float32)
                scale[mask] = 1.0
                color = color * scale[..., None]
                alpha = alpha * scale
                zbuffer = np.where(mask, np.minimum(zbuffer, layer.depth), zbuffer)
            else:
                continue

        # Proper front-to-back premultiplied composition
        one_minus_acc = (1.0 - accum_alpha)[..., None]
        accum_color = accum_color + color * one_minus_acc
        accum_alpha = accum_alpha + alpha * (1.0 - accum_alpha)

    accum_color = np.clip(accum_color, 0.0, 1.0)
    accum_alpha = np.clip(accum_alpha, 0.0, 1.0)
    return accum_color, accum_alpha


def render_frame(
    layers: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray, float, Optional[np.ndarray]]],
    Ks: np.ndarray,
    Kt: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    output_size: Tuple[int, int],
    edge_smooth_px: int = 2,
    use_zbuffer: bool = True,
    crack_filler: Optional[Callable[[Sequence[WarpedLayer]], Sequence[WarpedLayer]]] = None,
    dps_params: Optional[DPSParams] = None,
    force_identity_warp: bool = False,
) -> np.ndarray:
    """Render novel view given per-layer data and camera parameters."""
    warped_layers: List[WarpedLayer] = []
    for color, alpha, depth_map, plane_depth, normal in layers:
        plane_normal = normal if normal is not None else np.array([0.0, 0.0, 1.0], dtype=np.float32)
        if force_identity_warp:
            H = np.eye(3, dtype=np.float32)
        else:
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

    if crack_filler is not None:
        warped_layers = list(crack_filler(warped_layers))

    # Safety: if coverage after warping is extremely low, fall back to identity warp
    # to avoid black frames due to misconfigured intrinsics/poses.
    total_alpha = sum(float(np.sum(w.alpha > 1e-5)) for w in warped_layers)
    height, width = output_size[1], output_size[0]
    coverage = total_alpha / max(1.0, float(height * width))
    if coverage < 0.01:
        warped_layers = []
        I = np.eye(3, dtype=np.float32)
        for color, alpha, depth_map, plane_depth, normal in layers:
            warped = warp_layer(
                color_srgb=color,
                alpha=alpha,
                depth_map=depth_map,
                H=I,
                output_size=output_size,
                edge_smooth_px=edge_smooth_px,
            )
            warped_layers.append(warped)

    layer_views: List[LayerView] = []
    for idx, layer in enumerate(warped_layers):
        rgba = np.dstack((layer.color, layer.alpha)).astype(np.float32)
        mask = (layer.alpha > 1e-5) & np.isfinite(layer.depth)
        layer_views.append(LayerView(rgba=rgba, depth=layer.depth, mask=mask, id=idx))

    accum_color, accum_alpha = composite_front_to_back(
        warped_layers=warped_layers, output_size=output_size, use_zbuffer=use_zbuffer
    )
    composite_rgba = np.dstack((accum_color, accum_alpha)).astype(np.float32)

    if dps_params is not None:
        strip_rgba, strip_mask = build_dynamic_strip(layer_views, dps_params)
        composite_rgba = apply_dynamic_strip(composite_rgba, strip_rgba, strip_mask)

    rgb_linear = np.zeros_like(composite_rgba[..., :3], dtype=np.float32)
    nonzero = composite_rgba[..., 3] > 1e-5
    if np.any(nonzero):
        rgb_linear[nonzero] = composite_rgba[..., :3][nonzero] / composite_rgba[..., 3][nonzero, None]
    rgb_linear = np.clip(rgb_linear, 0.0, 1.0)
    srgb = linear_to_srgb(rgb_linear)
    srgb = np.clip(srgb * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(srgb, cv2.COLOR_RGB2BGR)
