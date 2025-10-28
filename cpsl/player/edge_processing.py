"""Edge processing utilities for sealing seams between CPSL layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import cv2
import numpy as np

from cpsl.config import CrackFixConfig
from cpsl.player.synth import WarpedLayer


_EPS = 1e-6


@dataclass
class CrackFiller:
    """Band-limited seam/crack repair for warped CPSL layers."""

    config: CrackFixConfig

    def apply(self, layers: Sequence[WarpedLayer]) -> List[WarpedLayer]:
        if not layers or not self.config.enabled:
            return list(layers)

        working = [WarpedLayer(layer.color.copy(), layer.alpha.copy(), layer.depth.copy()) for layer in layers]
        band_mask = self._build_edge_band(working)
        if not np.any(band_mask):
            return working

        self._dilate_foregrounds(working, band_mask)
        self._pull_background(working, band_mask)
        self._inpaint_residuals(working, band_mask)
        return working

    # ------------------------------------------------------------------ #
    def _build_edge_band(self, layers: Sequence[WarpedLayer]) -> np.ndarray:
        band = np.zeros_like(layers[0].alpha, dtype=bool)
        alpha_thresh = float(self.config.edge_alpha_threshold)
        depth_thresh = float(self.config.edge_depth_threshold)

        for layer in layers:
            alpha = layer.alpha.astype(np.float32)
            grad_ax = cv2.Sobel(alpha, cv2.CV_32F, 1, 0, ksize=3)
            grad_ay = cv2.Sobel(alpha, cv2.CV_32F, 0, 1, ksize=3)
            grad_a = cv2.magnitude(grad_ax, grad_ay)
            mask_alpha = grad_a > alpha_thresh

            depth = layer.depth.astype(np.float32)
            finite = np.isfinite(depth)
            if np.any(finite):
                depth_clean = depth.copy()
                depth_clean[~finite] = 0.0
                grad_dx = cv2.Sobel(depth_clean, cv2.CV_32F, 1, 0, ksize=3)
                grad_dy = cv2.Sobel(depth_clean, cv2.CV_32F, 0, 1, ksize=3)
                grad_d = cv2.magnitude(grad_dx, grad_dy)
                mask_depth = grad_d > depth_thresh
            else:
                mask_depth = np.zeros_like(mask_alpha, dtype=bool)

            band = np.logical_or(band, mask_alpha, out=band)
            band = np.logical_or(band, mask_depth, out=band)

        if self.config.band_px > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.config.band_px * 2 + 1, self.config.band_px * 2 + 1),
            )
            band = cv2.dilate(band.astype(np.uint8), kernel, iterations=1).astype(bool)

        return band

    # ------------------------------------------------------------------ #
    def _dilate_foregrounds(self, layers: Sequence[WarpedLayer], band_mask: np.ndarray) -> None:
        if self.config.dilate_px <= 0:
            return
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.config.dilate_px * 2 + 1, self.config.dilate_px * 2 + 1)
        )
        kernel_norm = kernel.astype(np.float32)

        for layer in layers:
            alpha = layer.alpha
            mask = alpha > _EPS
            if not np.any(mask):
                continue

            dilated = cv2.dilate(mask.astype(np.uint8), kernel)
            candidates = dilated.astype(bool)
            candidates &= ~mask
            candidates &= band_mask
            if not np.any(candidates):
                continue

            count = cv2.filter2D(mask.astype(np.float32), -1, kernel_norm, borderType=cv2.BORDER_REFLECT)
            count[count <= 0] = 1.0

            sum_alpha = cv2.filter2D(alpha, -1, kernel_norm, borderType=cv2.BORDER_REFLECT)
            avg_alpha = sum_alpha / count

            color = layer.color
            sum_color = np.stack(
                [
                    cv2.filter2D(color[:, :, channel], -1, kernel_norm, borderType=cv2.BORDER_REFLECT)
                    for channel in range(color.shape[2])
                ],
                axis=2,
            )
            avg_color = sum_color / count[..., None]

            depth = layer.depth
            depth_masked = depth.copy()
            depth_masked[~mask] = 0.0
            sum_depth = cv2.filter2D(depth_masked, -1, kernel_norm, borderType=cv2.BORDER_REFLECT)
            avg_depth = sum_depth / count

            valid_depth = depth[np.isfinite(depth)]
            if valid_depth.size == 0:
                depth_tol = np.inf
            else:
                depth_range = float(valid_depth.max() - valid_depth.min())
                depth_tol = max(self.config.depth_tol_rel * max(depth_range, _EPS), _EPS)

            depth_here = depth
            depth_compare = np.abs(avg_depth - depth_here)
            depth_gate = np.logical_or(~np.isfinite(depth_here), depth_compare <= depth_tol)

            write_mask = candidates & depth_gate
            if not np.any(write_mask):
                continue

            alpha_fill = np.clip(avg_alpha[write_mask], 0.0, 1.0)
            layer.alpha[write_mask] = np.clip(alpha_fill, 0.0, 1.0)
            layer.color[write_mask] = np.clip(avg_color[write_mask], 0.0, 1.0)
            layer.depth[write_mask] = avg_depth[write_mask]

    # ------------------------------------------------------------------ #
    def _pull_background(self, layers: Sequence[WarpedLayer], band_mask: np.ndarray) -> None:
        if self.config.fill_strength <= 0.0 or len(layers) < 2:
            return

        total = len(layers)
        fill_strength = float(self.config.fill_strength)
        depth_jump_rel = float(self.config.depth_jump_rel)
        offset = float(self.config.micro_offset_px)

        for idx in range(total - 1):
            front = layers[idx]
            deeper_stack = layers[idx + 1 :]
            if not deeper_stack:
                continue

            front_hole = (front.alpha < 1e-3) & band_mask
            if not np.any(front_hole):
                continue

            sample_x = sample_y = None
            if offset > _EPS:
                grad_x = cv2.Sobel(front.alpha.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(front.alpha.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
                norm = np.sqrt(grad_x**2 + grad_y**2)
                norm[norm < _EPS] = 1.0
                nx = -grad_x / norm
                ny = -grad_y / norm
                h, w = front.alpha.shape
                grid_x, grid_y = np.meshgrid(
                    np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32)
                )
                sample_x = np.clip(grid_x + nx * offset, 0.0, w - 1.0).astype(np.float32)
                sample_y = np.clip(grid_y + ny * offset, 0.0, h - 1.0).astype(np.float32)

            best_alpha = np.zeros_like(front.alpha)
            best_color = np.zeros_like(front.color)
            best_depth = np.full_like(front.depth, np.inf)

            for deeper in deeper_stack:
                if sample_x is not None:
                    sampled_alpha = cv2.remap(
                        deeper.alpha.astype(np.float32),
                        sample_x,
                        sample_y,
                        interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=0.0,
                    )
                    sampled_color = np.stack(
                        [
                            cv2.remap(
                                deeper.color[:, :, c].astype(np.float32),
                                sample_x,
                                sample_y,
                                interpolation=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT,
                            )
                            for c in range(deeper.color.shape[2])
                        ],
                        axis=2,
                    )
                    depth_src = deeper.depth.astype(np.float32)
                    depth_src[np.isinf(depth_src)] = np.finfo(np.float32).max
                    sampled_depth = cv2.remap(
                        depth_src,
                        sample_x,
                        sample_y,
                        interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REFLECT,
                    )
                    max_val = np.finfo(np.float32).max * 0.5
                    sampled_depth[sampled_depth >= max_val] = np.inf
                else:
                    sampled_alpha = deeper.alpha
                    sampled_color = deeper.color
                    sampled_depth = deeper.depth

                candidate_alpha = sampled_alpha * fill_strength
                mask = (candidate_alpha > 1e-3) & front_hole
                if not np.any(mask):
                    continue

                depth_diff = sampled_depth - front.depth
                depth_gate = np.logical_or(~np.isfinite(front.depth), depth_diff >= depth_jump_rel)
                mask &= depth_gate
                if not np.any(mask):
                    continue

                replace = candidate_alpha > best_alpha
                replace &= mask
                if not np.any(replace):
                    continue

                best_alpha[replace] = candidate_alpha[replace]
                best_color[replace] = sampled_color[replace]
                best_depth[replace] = sampled_depth[replace]

            valid = best_alpha > 0
            if not np.any(valid):
                continue

            residual = np.clip(1.0 - front.alpha[valid], 0.0, 1.0)
            applied_alpha = np.minimum(best_alpha[valid], residual)
            front.color[valid] += best_color[valid] * np.where(
                best_alpha[valid] > _EPS,
                applied_alpha / np.maximum(best_alpha[valid], _EPS),
                0.0,
            )[..., None]
            front.alpha[valid] += applied_alpha
            front.depth[valid] = np.where(
                np.isfinite(best_depth[valid]),
                np.minimum(best_depth[valid], front.depth[valid]),
                front.depth[valid],
            )
            front.alpha = np.clip(front.alpha, 0.0, 1.0)
            front.color = np.clip(front.color, 0.0, 1.0)

    # ------------------------------------------------------------------ #
    def _inpaint_residuals(self, layers: Sequence[WarpedLayer], band_mask: np.ndarray) -> None:
        if self.config.inpaint_alpha <= 0.0:
            return

        radius = max(1, int(self.config.inpaint_radius))
        fill_alpha = float(self.config.inpaint_alpha)

        for layer in layers:
            residual = band_mask & (layer.alpha < 1e-3)
            if not np.any(residual):
                continue

            rgb = np.zeros_like(layer.color)
            mask_nonzero = layer.alpha > _EPS
            if np.any(mask_nonzero):
                rgb[mask_nonzero] = layer.color[mask_nonzero] / np.clip(
                    layer.alpha[mask_nonzero][..., None], _EPS, None
                )
            rgb_bgr = cv2.cvtColor(np.clip(rgb, 0.0, 1.0), cv2.COLOR_RGB2BGR)
            inpaint_mask = residual.astype(np.uint8) * 255
            inpainted_bgr = cv2.inpaint(
                (rgb_bgr * 255.0).astype(np.uint8),
                inpaint_mask,
                radius,
                cv2.INPAINT_TELEA,
            )
            inpainted_rgb = cv2.cvtColor(inpainted_bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2RGB)

            alpha_inc = np.minimum(fill_alpha, 1.0 - layer.alpha[residual])
            layer.color[residual] += inpainted_rgb[residual] * alpha_inc[..., None]
            layer.alpha[residual] += alpha_inc

            finite_depth = layer.depth[np.isfinite(layer.depth)]
            fill_depth = float(np.median(finite_depth)) if finite_depth.size else 1.0
            layer.depth[residual] = fill_depth

            layer.color = np.clip(layer.color, 0.0, 1.0)
            layer.alpha = np.clip(layer.alpha, 0.0, 1.0)
