"""Depth-driven region clustering for CPSL layer extraction."""

from __future__ import annotations

import cv2
import numpy as np


def smooth_depth(depth: np.ndarray, kernel: int) -> np.ndarray:
    """Apply edge-preserving smoothing to the depth map."""
    if kernel <= 1:
        return depth
    radius = max(1, kernel | 1)  # ensure odd
    depth_normalized = cv2.normalize(depth, None, 0.0, 1.0, cv2.NORM_MINMAX)
    smoothed = cv2.bilateralFilter(depth_normalized.astype(np.float32), radius, 0.1, radius * 2)
    return smoothed


def cluster_depth_regions(
    depth_map: np.ndarray,
    target_layers: int,
    spatial_weight: float,
    min_area_ratio: float,
) -> np.ndarray:
    """Cluster pixels in (depth, x, y) feature space to obtain layer labels."""
    height, width = depth_map.shape
    total_pixels = float(height * width)

    xs, ys = np.meshgrid(np.linspace(-1.0, 1.0, width), np.linspace(-1.0, 1.0, height))
    features = np.stack(
        [
            depth_map.astype(np.float32).reshape(-1),
            (xs * spatial_weight).astype(np.float32).reshape(-1),
            (ys * spatial_weight).astype(np.float32).reshape(-1),
        ],
        axis=1,
    )

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-4)
    attempts = 5
    flags = cv2.KMEANS_PP_CENTERS
    _, labels, centers = cv2.kmeans(features, target_layers, None, criteria, attempts, flags)
    labels = labels.reshape((height, width))

    # Order clusters from nearest to farthest based on depth mean.
    depth_centers = centers[:, 0]
    order = np.argsort(depth_centers)
    remap = {int(old): int(new) for new, old in enumerate(order)}
    ordered_labels = np.vectorize(remap.get)(labels)

    # Remove extremely small regions by merging them with nearest depth cluster.
    cleaned = ordered_labels.copy()
    for label_id in range(target_layers):
        mask = ordered_labels == label_id
        area = float(mask.sum())
        if area / total_pixels >= min_area_ratio or label_id == target_layers - 1:
            continue
        neighbor = label_id + 1 if label_id + 1 < target_layers else label_id - 1
        cleaned[mask] = neighbor

    return cleaned

