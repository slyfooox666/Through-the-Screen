"""Metrics for evaluating CPSL crack-fix performance."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

_EPS = 1e-6


def crack_rate(alpha: np.ndarray, band_px: int = 5, alpha_threshold: float = 0.95) -> float:
    """Compute the fraction of pixels in the seam band with low composite alpha."""
    band = _edge_band(alpha, band_px)
    if band.sum() == 0:
        return 0.0
    cracks = (alpha < alpha_threshold) & band
    return float(cracks.sum() / max(band.sum(), 1))


def boundary_f_measure(
    pred_alpha: np.ndarray,
    gt_alpha: np.ndarray,
    band_px: int = 3,
) -> float:
    """Compute boundary F-score between prediction and ground truth trimaps."""
    pred_band = _edge_band(pred_alpha, band_px)
    gt_band = _edge_band(gt_alpha, band_px)

    tp = float(np.logical_and(pred_band, gt_band).sum())
    precision = tp / max(pred_band.sum(), 1)
    recall = tp / max(gt_band.sum(), 1)
    if precision + recall < _EPS:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def trimap_iou(pred_alpha: np.ndarray, gt_alpha: np.ndarray, width_px: int = 3) -> float:
    """Compute IoU inside a symmetric trimap band around GT boundaries."""
    gt_band = _edge_band(gt_alpha, width_px)
    if gt_band.sum() == 0:
        return 1.0
    pred_fg = pred_alpha >= 0.5
    gt_fg = gt_alpha >= 0.5
    intersection = np.logical_and(pred_fg, gt_fg) & gt_band
    union = np.logical_or(pred_fg, gt_fg) & gt_band
    return float(intersection.sum() / max(union.sum(), 1))


def _edge_band(alpha: np.ndarray, band_px: int) -> np.ndarray:
    alpha = alpha.astype(np.float32)
    grad_x = cv2.Sobel(alpha, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(alpha, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(grad_x, grad_y)
    band = grad > 1e-3
    if band_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (band_px * 2 + 1, band_px * 2 + 1))
        band = cv2.dilate(band.astype(np.uint8), kernel, iterations=1).astype(bool)
    return band
