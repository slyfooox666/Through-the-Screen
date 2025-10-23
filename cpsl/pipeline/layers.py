"""Layer generation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import cv2
import numpy as np


@dataclass
class LayerData:
    """Container for CPSL layer assets."""

    index: int
    class_name: str
    depth_mean: float
    depth_std: float
    rgba: np.ndarray
    alpha: np.ndarray
    depth_map: np.ndarray


def _soften_mask(mask: np.ndarray, sigma: float, boundary_band: int, alpha_max: float) -> np.ndarray:
    if boundary_band <= 0:
        return mask.astype(np.float32)
    kernel = int(boundary_band * 2 + 1)
    blurred = cv2.GaussianBlur(mask.astype(np.float32), (kernel, kernel), sigmaX=sigma)
    blurred = np.clip(blurred, 0.0, 1.0)
    return blurred * alpha_max


def generate_layers(
    frame_bgr: np.ndarray,
    depth_map: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str],
    target_layers: int,
    promote_classes: Iterable[str],
    boundary_band_px: int,
    soft_alpha_sigma: float,
    soft_alpha_max: float,
    min_area_ratio: float = 0.005,
) -> List[LayerData]:
    """Generate CPSL layers by grouping pixels according to depth-aware clusters."""
    height, width = labels.shape
    total_pixels = height * width
    promote_classes = set(promote_classes)
    candidate_layers: List[Dict[str, object]] = []

    for class_idx in np.unique(labels):
        mask = labels == class_idx
        area = mask.sum()
        if area / total_pixels < min_area_ratio:
            continue
        depth_values = depth_map[mask]
        if depth_values.size == 0:
            continue
        depth_mean = float(depth_values.mean())
        depth_std = float(depth_values.std())
        class_name = class_names[class_idx] if class_idx < len(class_names) else f"class_{class_idx}"
        priority = depth_mean
        if class_name in promote_classes:
            priority -= 0.1  # pull slightly forward
        candidate_layers.append(
            {
                "class_idx": int(class_idx),
                "class_name": class_name,
                "mask": mask,
                "depth_mean": depth_mean,
                "depth_std": depth_std,
                "priority": priority,
            }
        )

    # Always add a background layer
    background_mask = np.ones_like(labels, dtype=bool)
    for cand in candidate_layers:
        background_mask &= ~cand["mask"]
    if background_mask.sum() > 0:
        depth_values = depth_map[background_mask]
        candidate_layers.append(
            {
                "class_idx": -1,
                "class_name": "background",
                "mask": background_mask,
                "depth_mean": float(depth_values.mean() if depth_values.size else 1.0),
                "depth_std": float(depth_values.std() if depth_values.size else 0.0),
                "priority": float(depth_values.mean() if depth_values.size else 1.0),
            }
        )

    candidate_layers.sort(key=lambda item: item["priority"])

    selected = candidate_layers[:target_layers]
    if len(candidate_layers) > target_layers:
        # Merge remaining into background
        merged_mask = np.zeros_like(labels, dtype=bool)
        depth_values = []
        for cand in candidate_layers[target_layers:]:
            merged_mask |= cand["mask"]
            depth_values.append(depth_map[cand["mask"]])
        if depth_values:
            merged_depth = np.concatenate(depth_values)
            selected.append(
                {
                    "class_idx": -2,
                    "class_name": "residual",
                    "mask": merged_mask,
                    "depth_mean": float(merged_depth.mean()),
                    "depth_std": float(merged_depth.std()),
                    "priority": float(merged_depth.mean()),
                }
            )

    layers: List[LayerData] = []
    frame_float = frame_bgr.astype(np.float32) / 255.0
    for idx, cand in enumerate(selected):
        mask = cand["mask"]
        soft_alpha = _soften_mask(mask, soft_alpha_sigma, boundary_band_px, soft_alpha_max)
        alpha_3c = np.repeat(soft_alpha[:, :, None], 3, axis=2)
        rgba = np.dstack((frame_float * alpha_3c, soft_alpha))
        depth_layer = np.where(mask, depth_map, 1.0).astype(np.float32)
        layers.append(
            LayerData(
                index=idx,
                class_name=cand["class_name"],
                depth_mean=float(cand["depth_mean"]),
                depth_std=float(cand["depth_std"]),
                rgba=rgba,
                alpha=soft_alpha,
                depth_map=depth_layer,
            )
        )
    return layers

