"""Dynamic Pixel Strip (DPS) seam filling for CPSL playback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from cpsl.utils import morph

_EPS = 1e-6


@dataclass
class LayerView:
    """Per-layer warped assets in linear premultiplied space."""

    rgba: np.ndarray  # HxWx4 float32 premultiplied (linear)
    depth: np.ndarray  # HxW float32
    mask: np.ndarray  # HxW bool
    id: int


@dataclass
class DpsConfig:
    """Configuration for the geometry-aware DPS strip."""

    band_px: int = 3
    feather_px: int = 2
    max_pull_px: int = 2  # legacy knob, retained for compatibility
    depth_tolerance: float = 0.02
    alpha_threshold: float = 0.98
    feather_weight: float = 0.65
    temporal_ema: float = 0.35
    extension_px: float = 1.5
    inpaint_radius: int = 2
    inpaint: str = "telea"
    backend: str = "auto"
    z_sigma: float = 0.01  # unused but kept for config compatibility
    color_sigma: float = 0.1  # unused but kept for config compatibility
    z_conf_thresh: float = 0.0  # unused but kept for config compatibility


@dataclass
class DPSParams(DpsConfig):
    """Backwards-compatible parameter surface used by playback configs."""

    def to_config(self) -> DpsConfig:
        return DpsConfig(
            band_px=self.band_px,
            feather_px=self.feather_px,
            max_pull_px=self.max_pull_px,
            depth_tolerance=self.depth_tolerance,
            alpha_threshold=self.alpha_threshold,
            feather_weight=self.feather_weight,
            temporal_ema=self.temporal_ema,
            extension_px=self.extension_px,
            inpaint_radius=self.inpaint_radius,
            inpaint=self.inpaint,
            backend=self.backend,
            z_sigma=self.z_sigma,
            color_sigma=self.color_sigma,
            z_conf_thresh=self.z_conf_thresh,
        )

    @classmethod
    def from_config(cls, cfg: "DPSConfigLike") -> "DPSParams":
        return cls(
            band_px=cfg.band_px,
            feather_px=getattr(cfg, "feather_px", cls.__dataclass_fields__["feather_px"].default),
            max_pull_px=getattr(cfg, "max_pull_px", cls.__dataclass_fields__["max_pull_px"].default),
            depth_tolerance=getattr(cfg, "depth_tolerance", cls.__dataclass_fields__["depth_tolerance"].default),
            alpha_threshold=getattr(cfg, "alpha_threshold", cls.__dataclass_fields__["alpha_threshold"].default),
            feather_weight=getattr(cfg, "feather_weight", cls.__dataclass_fields__["feather_weight"].default),
            temporal_ema=getattr(cfg, "temporal_ema", cls.__dataclass_fields__["temporal_ema"].default),
            extension_px=getattr(cfg, "extension_px", cls.__dataclass_fields__["extension_px"].default),
            inpaint_radius=getattr(cfg, "inpaint_radius", cls.__dataclass_fields__["inpaint_radius"].default),
            inpaint=getattr(cfg, "inpaint", cls.__dataclass_fields__["inpaint"].default),
            backend=getattr(cfg, "backend", cls.__dataclass_fields__["backend"].default),
            z_sigma=getattr(cfg, "z_sigma", cls.__dataclass_fields__["z_sigma"].default),
            color_sigma=getattr(cfg, "color_sigma", cls.__dataclass_fields__["color_sigma"].default),
            z_conf_thresh=getattr(cfg, "z_conf_thresh", cls.__dataclass_fields__["z_conf_thresh"].default),
        )


class DPSConfigLike:
    """Protocol-like helper for static type checking."""

    band_px: int
    feather_px: int
    max_pull_px: int
    depth_tolerance: float
    alpha_threshold: float
    feather_weight: float
    temporal_ema: float
    extension_px: float
    inpaint_radius: int
    inpaint: str
    backend: str
    z_sigma: float
    color_sigma: float
    z_conf_thresh: float


@dataclass
class _TemporalState:
    strip_rgba: Optional[np.ndarray] = None
    strip_mask: Optional[np.ndarray] = None


_STATE = _TemporalState()


def reset_dps_state() -> None:
    """Reset the temporal EMA history."""
    _STATE.strip_rgba = None
    _STATE.strip_mask = None


def _gather_channel(stack: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return np.take_along_axis(stack, indices[None, ...], axis=0)[0]


def _ensure_alpha(alpha: np.ndarray) -> np.ndarray:
    if alpha.ndim == 4:
        return alpha[..., 0]
    return alpha


def _remap_multi(array: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    """Bilinear remap of a multi-channel image."""
    channels: List[np.ndarray] = []
    for ch in range(array.shape[2]):
        channels.append(
            cv2.remap(
                array[..., ch],
                map_x,
                map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT101,
            )
        )
    return np.stack(channels, axis=-1)


def _remap_single(array: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    return cv2.remap(
        array,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )


def _compute_seam_mask(alpha_stack: np.ndarray, params: DpsConfig) -> np.ndarray:
    coverage = alpha_stack.sum(axis=0)
    uncovered = coverage < params.alpha_threshold

    height, width = alpha_stack.shape[1:3]
    edges = np.zeros((height, width), dtype=bool)
    edge_thresh = 0.02
    for alpha in alpha_stack:
        grad_x = cv2.Sobel(alpha, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(alpha, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(grad_x, grad_y)
        edges |= magnitude > edge_thresh

    seam = uncovered | edges
    seam = morph.dilate(seam, params.band_px, backend=params.backend)
    return seam


def _classify_pixels(
    fg_depth: np.ndarray,
    bg_depth: np.ndarray,
    fg_valid: np.ndarray,
    bg_valid: np.ndarray,
    seam_mask: np.ndarray,
    params: DpsConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return boolean masks for fg-extension, bg-reveal, and orphan seam pixels."""
    depth_gap = np.clip(bg_depth - fg_depth, 0.0, None)
    denom = np.maximum(np.abs(fg_depth), params.depth_tolerance)
    rel_gap = np.where(bg_valid, depth_gap / denom, np.inf)

    fg_extension = seam_mask & fg_valid & ((rel_gap <= params.depth_tolerance) | ~bg_valid)
    bg_reveal = seam_mask & bg_valid & (rel_gap > params.depth_tolerance)
    orphan = seam_mask & ~(fg_extension | bg_reveal)
    return fg_extension, bg_reveal, orphan


def _extend_foreground(
    color_stack: np.ndarray,
    alpha_stack: np.ndarray,
    fg_idx: np.ndarray,
    fg_mask: np.ndarray,
    params: DpsConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    height, width = fg_mask.shape
    fg_color = np.zeros((height, width, 3), dtype=np.float32)
    fg_alpha = np.zeros((height, width), dtype=np.float32)
    if not np.any(fg_mask):
        return fg_color, fg_alpha

    grid_y, grid_x = np.meshgrid(np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32), indexing="ij")

    unique_ids = np.unique(fg_idx[fg_mask])
    for layer_id in unique_ids:
        layer_mask = fg_mask & (fg_idx == layer_id)
        if not np.any(layer_mask):
            continue
        layer_alpha = alpha_stack[layer_id]
        layer_color = color_stack[layer_id]
        valid_region = layer_alpha > 1e-5
        if not np.any(valid_region):
            continue

        dist = cv2.distanceTransform((~valid_region).astype(np.uint8), cv2.DIST_L2, 3).astype(np.float32)
        grad_y, grad_x = np.gradient(dist)
        norm = np.sqrt(grad_x * grad_x + grad_y * grad_y) + _EPS
        dir_x = -grad_x / norm
        dir_y = -grad_y / norm

        offset = float(max(params.extension_px, 1.0))
        map_x = np.clip(grid_x + dir_x * offset, 0.0, width - 1.0).astype(np.float32)
        map_y = np.clip(grid_y + dir_y * offset, 0.0, height - 1.0).astype(np.float32)

        sampled_color = _remap_multi(layer_color, map_x, map_y)
        sampled_alpha = _remap_single(layer_alpha, map_x, map_y)

        fg_color[layer_mask] = sampled_color[layer_mask]
        fg_alpha[layer_mask] = sampled_alpha[layer_mask]

    return fg_color, fg_alpha


def _reveal_background(
    color_stack: np.ndarray,
    alpha_stack: np.ndarray,
    bg_idx: np.ndarray,
    bg_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    height, width = bg_mask.shape
    bg_color = np.zeros((height, width, 3), dtype=np.float32)
    bg_alpha = np.zeros((height, width), dtype=np.float32)
    if not np.any(bg_mask):
        return bg_color, bg_alpha

    unique_ids = np.unique(bg_idx[bg_mask])
    for layer_id in unique_ids:
        layer_mask = bg_mask & (bg_idx == layer_id)
        if not np.any(layer_mask):
            continue
        layer_color = color_stack[layer_id]
        layer_alpha = alpha_stack[layer_id]
        bg_color[layer_mask] = layer_color[layer_mask]
        bg_alpha[layer_mask] = layer_alpha[layer_mask]
    return bg_color, bg_alpha


def _apply_temporal_smoothing(
    strip_rgba: np.ndarray,
    strip_mask: np.ndarray,
    params: DpsConfig,
    D_star: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    if _STATE.strip_rgba is None or params.temporal_ema <= 0.0:
        return strip_rgba, strip_mask

    prev_rgba = _STATE.strip_rgba
    prev_mask = _STATE.strip_mask if _STATE.strip_mask is not None else np.zeros_like(strip_mask)

    height, width = strip_mask.shape
    grid_y, grid_x = np.meshgrid(
        np.arange(height, dtype=np.float32),
        np.arange(width, dtype=np.float32),
        indexing="ij",
    )

    if D_star is None:
        warped_rgba = prev_rgba
        warped_mask = prev_mask
    elif D_star.shape == (3, 3):
        warped_rgba = cv2.warpPerspective(
            prev_rgba,
            D_star,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
        warped_mask = cv2.warpPerspective(
            prev_mask.astype(np.uint8),
            D_star,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ) > 0
    elif D_star.ndim == 3 and D_star.shape[2] == 2:
        map_x = grid_x + D_star[..., 0].astype(np.float32)
        map_y = grid_y + D_star[..., 1].astype(np.float32)
        warped_rgba = _remap_multi(prev_rgba, map_x, map_y)
        warped_mask = cv2.remap(
            prev_mask.astype(np.float32),
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        ) > 0.5
    else:
        raise ValueError("Unsupported D_star shape for DPS temporal smoothing.")

    ema = np.clip(params.temporal_ema, 0.0, 1.0)
    union_mask = strip_mask | warped_mask
    if not np.any(union_mask):
        return strip_rgba, strip_mask

    blended = strip_rgba.copy()
    blended[union_mask, :3] = (
        (1.0 - ema) * strip_rgba[union_mask, :3] + ema * warped_rgba[union_mask, :3]
    )
    blended[union_mask, 3] = (
        (1.0 - ema) * strip_rgba[union_mask, 3] + ema * warped_rgba[union_mask, 3]
    )
    return blended, union_mask


def _band_limited_inpaint(
    strip_rgba: np.ndarray,
    residual_mask: np.ndarray,
    params: DpsConfig,
) -> None:
    if params.inpaint_radius <= 0 or not np.any(residual_mask):
        return

    mask_uint8 = residual_mask.astype(np.uint8) * 255
    alpha = strip_rgba[..., 3]
    rgb = np.zeros_like(strip_rgba[..., :3], dtype=np.float32)
    valid = alpha > _EPS
    rgb[valid] = strip_rgba[..., :3][valid] / np.maximum(alpha[valid, None], _EPS)
    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    method = params.inpaint.lower()
    if method not in {"telea", "ns"}:
        method = "telea"
    flag = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
    rgb_uint8 = np.clip(rgb_bgr * 255.0, 0.0, 255.0).astype(np.uint8)
    inpainted = cv2.inpaint(rgb_uint8, mask_uint8, params.inpaint_radius, flag)
    inpainted_rgb = cv2.cvtColor(inpainted.astype(np.float32) / 255.0, cv2.COLOR_BGR2RGB)

    strip_rgba[residual_mask, :3] = inpainted_rgb[residual_mask] * params.feather_weight
    strip_rgba[residual_mask, 3] = np.clip(params.feather_weight, 0.0, 1.0)


def apply_dps(
    warped: Dict[str, np.ndarray],
    D_star: Optional[np.ndarray] = None,
    cfg: Optional[DpsConfig] = None,
) -> Dict[str, np.ndarray]:
    """
    Detect and fill cracks between CPSL layers using geometry-aware feathering.

    Parameters
    ----------
    warped:
        Dictionary containing per-layer premultiplied colour `color` (KxHxWx3),
        `alpha` (KxHxW or KxHxWx1), and `depth` (KxHxW).
    D_star:
        Optional temporal reprojection map (2-channel flow or 3x3 homography)
        used to warp the previous strip for EMA smoothing.
    cfg:
        Optional configuration override.
    """

    params = cfg if cfg is not None else DpsConfig()
    color_stack = warped["color"].astype(np.float32)
    alpha_stack = _ensure_alpha(warped["alpha"].astype(np.float32))
    depth_stack = warped["depth"].astype(np.float32)

    if color_stack.shape[:3] != alpha_stack.shape:
        raise ValueError("color and alpha stacks must share dimensions (K,H,W).")

    seam_mask = _compute_seam_mask(alpha_stack, params)
    if not np.any(seam_mask):
        strip_rgba = np.zeros(seam_mask.shape + (4,), dtype=np.float32)
        return {"strip_rgba": strip_rgba, "strip_mask": seam_mask}

    depth_mask = (alpha_stack > 1e-5) & np.isfinite(depth_stack)
    depth_for_sort = np.where(depth_mask, depth_stack, np.inf)
    sorted_idx = np.argsort(depth_for_sort, axis=0)
    valid_counts = depth_mask.sum(axis=0)

    fg_idx = sorted_idx[0]
    fg_valid = valid_counts > 0
    bg_idx = np.where(valid_counts > 1, sorted_idx[1], fg_idx)
    bg_valid = valid_counts > 1

    fg_depth = _gather_channel(depth_stack, fg_idx)
    bg_depth = _gather_channel(depth_stack, bg_idx)
    fg_alpha = _gather_channel(alpha_stack, fg_idx)
    bg_alpha = _gather_channel(alpha_stack, bg_idx)

    fg_depth[~fg_valid] = np.inf
    fg_alpha[~fg_valid] = 0.0
    bg_depth[~bg_valid] = np.inf
    bg_alpha[~bg_valid] = 0.0

    fg_extension_mask, bg_reveal_mask, orphan_mask = _classify_pixels(
        fg_depth,
        bg_depth,
        fg_valid,
        bg_valid,
        seam_mask,
        params,
    )

    fg_ext_color, fg_ext_alpha = _extend_foreground(
        color_stack,
        alpha_stack,
        fg_idx,
        fg_extension_mask,
        params,
    )
    bg_rev_color, bg_rev_alpha = _reveal_background(
        color_stack,
        alpha_stack,
        bg_idx,
        bg_reveal_mask,
    )

    feather = morph.feather_mask(seam_mask, params.feather_px).astype(np.float32)
    feather = np.clip(feather, 0.0, 1.0)

    strip_rgba = np.zeros(seam_mask.shape + (4,), dtype=np.float32)

    if np.any(fg_extension_mask):
        fg_alpha_base = np.maximum(fg_ext_alpha, 0.0)
        alpha_scale = feather[fg_extension_mask] * params.feather_weight
        alpha_target = np.clip(fg_alpha_base[fg_extension_mask] * alpha_scale, 0.0, 1.0)
        color_scale = alpha_target / np.maximum(fg_alpha_base[fg_extension_mask], _EPS)
        strip_rgba[fg_extension_mask, :3] = fg_ext_color[fg_extension_mask] * color_scale[:, None]
        strip_rgba[fg_extension_mask, 3] = alpha_target

    if np.any(bg_reveal_mask):
        fg_alpha_local = fg_alpha[bg_reveal_mask]
        alpha_budget = np.clip(1.0 - fg_alpha_local, 0.0, 1.0)
        alpha_target = np.clip(
            bg_rev_alpha[bg_reveal_mask] * alpha_budget * params.feather_weight * feather[bg_reveal_mask],
            0.0,
            1.0,
        )
        color_scale = alpha_target / np.maximum(bg_rev_alpha[bg_reveal_mask], _EPS)
        strip_rgba[bg_reveal_mask, :3] = bg_rev_color[bg_reveal_mask] * color_scale[:, None]
        strip_rgba[bg_reveal_mask, 3] = alpha_target

    residual_mask = seam_mask & (strip_rgba[..., 3] < 1e-4)
    _band_limited_inpaint(strip_rgba, residual_mask, params)

    strip_rgba, final_mask = _apply_temporal_smoothing(strip_rgba, seam_mask, params, D_star)

    strip_rgba[~final_mask] = 0.0
    strip_mask = final_mask & (strip_rgba[..., 3] > 1e-5)

    _STATE.strip_rgba = strip_rgba.copy()
    _STATE.strip_mask = strip_mask.copy()

    return {
        "strip_rgba": strip_rgba,
        "strip_mask": strip_mask,
        "fg_extension_mask": fg_extension_mask,
        "bg_reveal_mask": bg_reveal_mask,
    }


def build_dynamic_strip(layers: List[LayerView], params: DPSParams) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct the dynamic pixel strip covering cracks between layers.

    Returns premultiplied strip RGBA and a boolean mask.
    """
    if not layers:
        raise ValueError("build_dynamic_strip requires at least one layer.")

    stack_color = np.stack([lv.rgba[..., :3] for lv in layers], axis=0).astype(np.float32)
    stack_alpha = np.stack([lv.rgba[..., 3] for lv in layers], axis=0).astype(np.float32)
    stack_depth = np.stack([lv.depth for lv in layers], axis=0).astype(np.float32)

    result = apply_dps(
        {"color": stack_color, "alpha": stack_alpha, "depth": stack_depth},
        cfg=params.to_config(),
    )
    return result["strip_rgba"], result["strip_mask"]


def apply_dynamic_strip(
    base_rgba: np.ndarray,
    strip_rgba: np.ndarray,
    strip_mask: np.ndarray,
) -> np.ndarray:
    """Composite the dynamic strip over the base premultiplied image."""
    if base_rgba.shape[2] != 4 or strip_rgba.shape[2] != 4:
        raise ValueError("apply_dynamic_strip expects HxWx4 inputs.")

    output = base_rgba.copy().astype(np.float32)
    mask = strip_mask
    if not np.any(mask):
        return output

    strip_alpha = strip_rgba[..., 3]
    inv_alpha = 1.0 - strip_alpha

    output_rgb = output[..., :3]
    output_alpha = output[..., 3]

    output_rgb[mask] = strip_rgba[mask, :3] + output_rgb[mask] * inv_alpha[mask, None]
    output_alpha[mask] = strip_alpha[mask] + output_alpha[mask] * inv_alpha[mask]

    output[..., :3] = np.clip(output_rgb, 0.0, 1.0)
    output[..., 3] = np.clip(output_alpha, 0.0, 1.0)
    return output
