"""Generate multi-layer volumetric frame sets from RGB or RGB-D imagery."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

try:
    import torch
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("MultiLayerGenerator requires PyTorch to be installed.") from exc


ImageLike = Union[str, Path, Image.Image, np.ndarray]
DepthLike = Optional[np.ndarray]


DEFAULT_SEGMENTATION_MODEL = "deeplabv3_mobilenet_v3_large"
_DEFAULT_SEG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_DEFAULT_SEG_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class Layer:
    """One semi-transparent slice of the layered representation."""

    name: str
    rgba: np.ndarray  # H × W × 4 in [0, 1]
    depth: np.ndarray  # H × W float32 depth map (NaN where inactive)
    mask: np.ndarray  # H × W bool mask of confident support
    statistics: Dict[str, float] = field(default_factory=dict)

    def save_debug_png(self, directory: Union[str, Path]) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        rgba_img = np.clip(self.rgba * 255.0, 0.0, 255.0).astype(np.uint8)
        Image.fromarray(rgba_img, mode="RGBA").save(directory / f"{self.name}_rgba.png")

        depth_vis = self.depth.copy()
        finite = np.isfinite(depth_vis)
        if finite.any():
            lo, hi = depth_vis[finite].min(), depth_vis[finite].max()
            if math.isclose(lo, hi):
                depth_vis = np.zeros_like(depth_vis)
            else:
                depth_vis = (depth_vis - lo) / (hi - lo)
        else:
            depth_vis = np.zeros_like(depth_vis)
        Image.fromarray(np.clip(depth_vis * 255.0, 0.0, 255.0).astype(np.uint8), mode="L").save(
            directory / f"{self.name}_depth.png"
        )

        with (directory / f"{self.name}_meta.json").open("w", encoding="utf-8") as fout:
            json.dump(self.statistics, fout, indent=2)


@dataclass
class LayeredFrameSet:
    """Collection of ordered layers representing a single frame."""

    source_image_path: Optional[Path]
    layers: List[Layer]  # index 0 = farthest, last = nearest
    depth_metadata: Dict[str, float]

    def composite(self) -> np.ndarray:
        if not self.layers:
            raise ValueError("No layers available to composite.")

        canvas = np.zeros_like(self.layers[0].rgba)
        for layer in self.layers:
            src_rgb = layer.rgba[..., :3]
            src_a = layer.rgba[..., 3:4]
            dst_rgb = canvas[..., :3]
            dst_a = canvas[..., 3:4]

            out_rgb = src_rgb * src_a + dst_rgb * (1.0 - src_a)
            out_a = src_a + dst_a * (1.0 - src_a)

            canvas[..., :3] = out_rgb
            canvas[..., 3:4] = np.clip(out_a, 0.0, 1.0)
        return canvas


class MultiLayerGenerator:
    """Convert RGB / RGB-D frames into layered RGBA depth sets."""

    def __init__(
        self,
        num_layers: int = 4,
        depth_model_type: str = "DPT_Large",
        device: Optional[Union[str, torch.device]] = None,
        edge_softening_px: float = 1.5,
        mask_threshold: float = 0.01,
        fill_occlusions: bool = True,
        occlusion_fill_iterations: int = 32,
        quantile_padding: float = 0.0,
        depth_normalisation: str = "percentile",
        depth_smoothing_sigma: float = 1.0,
        mask_morphology_radius: int = 1,
        segmentation_model_type: Optional[str] = DEFAULT_SEGMENTATION_MODEL,
        semantic_confidence_threshold: float = 0.25,
        semantic_depth_std_threshold: float = 0.08,
    ) -> None:
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.num_layers = num_layers
        self.depth_model_type = depth_model_type
        self.device = torch.device(device) if device is not None else self._auto_device()
        self.edge_softening_px = max(0.0, edge_softening_px)
        self.mask_threshold = mask_threshold
        self.fill_occlusions = fill_occlusions
        self.occlusion_fill_iterations = occlusion_fill_iterations
        self.quantile_padding = quantile_padding
        self.depth_normalisation = depth_normalisation
        self.depth_smoothing_sigma = max(0.0, depth_smoothing_sigma)
        self.mask_morphology_radius = max(0, mask_morphology_radius)

        self.segmentation_model_type = segmentation_model_type
        self.semantic_confidence_threshold = float(max(0.0, semantic_confidence_threshold))
        self.semantic_depth_std_threshold = float(max(0.0, semantic_depth_std_threshold))
        self._semantics_enabled = segmentation_model_type is not None

        self._midas_model = None
        self._midas_transform = None
        self._gaussian_kernel_cache: Dict[Tuple[int, float], torch.Tensor] = {}
        self._segmentation_model = None
        self._segmentation_norm_mean: Optional[torch.Tensor] = None
        self._segmentation_norm_std: Optional[torch.Tensor] = None
        self._prev_rgb: Optional[np.ndarray] = None
        self._prev_semantic_labels: Optional[np.ndarray] = None
        self._prev_semantic_scores: Optional[np.ndarray] = None
        self._prev_semantic_source: Optional[str] = None

    # ------------------------------------------------------------------
    def generate(
        self,
        image: ImageLike,
        depth_map: DepthLike = None,
        num_layers: Optional[int] = None,
        semantic_map: Optional[np.ndarray] = None,
    ) -> LayeredFrameSet:
        pil_image, image_path = self._load_image(image)
        rgb = np.asarray(pil_image.convert("RGB"), dtype=np.float32) / 255.0

        if depth_map is None:
            depth_map = self._estimate_depth(pil_image)
        else:
            depth_map = np.asarray(depth_map, dtype=np.float32)
            if depth_map.ndim == 3:
                depth_map = depth_map[..., 0]
            if depth_map.shape != rgb.shape[:2]:
                raise ValueError("Depth map shape must match RGB dimensions.")

        depth_map = self._normalise_depth(depth_map)
        # Lightly denoise depth to stabilise subsequent quantile binning.
        depth_map = self._smooth_depth(depth_map)

        semantic_guidance = None
        if self._semantics_enabled:
            semantic_guidance = self._semantic_guidance(pil_image, rgb, semantic_map)
            if semantic_guidance is not None and semantic_guidance["labels"].shape != depth_map.shape:
                raise ValueError("Semantic map shape must match RGB dimensions.")

        target_layers = num_layers or self.num_layers
        masks, quantiles = self._stratify_depth(depth_map, target_layers)
        if semantic_guidance is not None:
            masks = self._blend_semantics(depth_map, masks, quantiles, semantic_guidance)
        specs = self._prepare_layer_specs(masks, depth_map)

        layers: List[Layer] = []
        occlusion = np.zeros_like(depth_map, dtype=bool)
        for spec in specs:  # nearest → farthest
            mask = spec["mask"]
            if mask.mean() < self.mask_threshold:
                continue

            layer_rgb = rgb.copy()
            depth_snapshot = np.where(mask, depth_map, np.nan)

            if self.fill_occlusions and occlusion.any():
                holes = occlusion & (~mask)
                if holes.any():
                    layer_rgb = self._progressive_inpaint(layer_rgb, holes)

            alpha = self._build_alpha(mask.astype(np.float32))
            rgba = np.dstack([layer_rgb, alpha])
            stats = {
                "depth_min": float(spec["depth_range"][0]),
                "depth_max": float(spec["depth_range"][1]),
                "depth_median": float(spec["depth_median"]),
                "coverage_ratio": float(mask.mean()),
            }
            stats.update(self._layer_semantic_stats(mask, semantic_guidance))
            layer = Layer(
                name=spec["name"],
                rgba=rgba,
                depth=depth_snapshot,
                mask=mask,
                statistics=stats,
            )
            layers.insert(0, layer)  # Ensure farthest layer ends up at index 0

            occlusion |= mask

        depth_meta = {
            "model": self.depth_model_type,
            "normalisation": self.depth_normalisation,
            "quantile_bounds": [float(q) for q in quantiles],
        }
        if semantic_guidance is not None:
            depth_meta["semantics"] = {
                "model": self.segmentation_model_type,
                "confidence_threshold": self.semantic_confidence_threshold,
                "depth_std_threshold": self.semantic_depth_std_threshold,
                "source": semantic_guidance.get("source"),
            }
        return LayeredFrameSet(image_path, layers, depth_meta)

    def generate_from_file(
        self,
        image_path: Union[str, Path],
        depth_map_path: Optional[Union[str, Path]] = None,
        num_layers: Optional[int] = None,
        semantic_map_path: Optional[Union[str, Path]] = None,
    ) -> LayeredFrameSet:
        depth = None
        if depth_map_path is not None:
            depth_path = Path(depth_map_path)
            if depth_path.suffix == ".npy":
                depth = np.load(depth_path)
            else:
                depth = np.asarray(Image.open(depth_path))
        semantic = None
        if semantic_map_path is not None:
            semantic_path = Path(semantic_map_path)
            if semantic_path.suffix == ".npy":
                semantic = np.load(semantic_path)
            else:
                semantic = np.asarray(Image.open(semantic_path))
        return self.generate(image_path, depth_map=depth, num_layers=num_layers, semantic_map=semantic)

    def to_layers(self, frame_bgr: np.ndarray) -> LayeredFrameSet:
        """Create a :class:`LayeredFrameSet` from an OpenCV BGR frame."""

        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("Expected an H×W×3 BGR frame")
        rgb = frame_bgr[..., ::-1].astype(np.uint8)
        return self.generate(rgb)

    # ------------------------------------------------------------------
    def _estimate_depth(self, image: Image.Image) -> np.ndarray:
        model, transform = self._load_midas()

        transformed = transform(np.asarray(image))
        tensor = transformed["image"] if isinstance(transformed, dict) else transformed
        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        tensor = tensor.to(self.device, dtype=torch.float32)

        with torch.no_grad():
            prediction = model(tensor)
            prediction = F.interpolate(
                prediction.unsqueeze(1),
                size=image.size[::-1],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        return prediction.cpu().numpy().astype(np.float32)

    def _normalise_depth(self, depth: np.ndarray) -> np.ndarray:
        depth = np.asarray(depth, dtype=np.float32)
        finite = np.isfinite(depth)
        if not finite.any():
            raise ValueError("Depth map contains no finite values.")

        if self.depth_normalisation == "percentile":
            lo, hi = np.percentile(depth[finite], [2.0, 98.0])
            depth = np.clip(depth, lo, hi)
        elif self.depth_normalisation == "minmax":
            lo, hi = float(np.nanmin(depth)), float(np.nanmax(depth))
        else:
            raise ValueError(f"Unsupported depth normalisation '{self.depth_normalisation}'.")

        denom = hi - lo
        if math.isclose(denom, 0.0):
            return np.zeros_like(depth)
        return (depth - lo) / denom

    def _stratify_depth(
        self, depth: np.ndarray, num_layers: int
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        quantiles = np.linspace(
            0.0 + self.quantile_padding,
            1.0 - self.quantile_padding,
            num_layers + 1,
        )
        quantiles[0], quantiles[-1] = 0.0, 1.0
        bounds = np.quantile(depth[np.isfinite(depth)], quantiles)

        masks: List[np.ndarray] = []
        for idx in range(num_layers):
            lower, upper = bounds[idx], bounds[idx + 1]
            if idx == num_layers - 1:
                mask = depth >= lower
            else:
                mask = (depth >= lower) & (depth < upper)
            masks.append(mask)
        return masks, bounds

    def _prepare_layer_specs(
        self, masks: Sequence[np.ndarray], depth: np.ndarray
    ) -> List[Dict[str, Union[str, float, np.ndarray]]]:
        specs: List[Dict[str, Union[str, float, np.ndarray]]] = []
        for idx, mask in enumerate(masks):
            if not mask.any():
                continue
            # Close small voids in the binary bin to avoid sparkly edges.
            mask = self._refine_mask(mask)
            if not mask.any():
                continue
            masked_depth = depth[mask]
            masked_depth = masked_depth[np.isfinite(masked_depth)]
            if masked_depth.size == 0:
                continue
            specs.append(
                {
                    "name": f"layer_{idx:02d}",
                    "mask": mask,
                    "depth_range": (float(np.min(masked_depth)), float(np.max(masked_depth))),
                    "depth_median": float(np.median(masked_depth)),
                }
            )
        specs.sort(key=lambda item: item["depth_median"], reverse=True)
        return specs

    # ------------------------------------------------------------------
    # Semantic guidance helpers
    # ------------------------------------------------------------------

    def _semantic_guidance(
        self,
        image: Image.Image,
        rgb: np.ndarray,
        semantic_map: Optional[np.ndarray],
    ) -> Optional[Dict[str, Any]]:
        if not self._semantics_enabled:
            return None

        labels: Optional[np.ndarray] = None
        scores: Optional[np.ndarray] = None
        source = None

        if semantic_map is not None:
            labels = np.asarray(semantic_map)
            if labels.ndim == 3:
                labels = labels[..., 0]
            labels = labels.astype(np.int32)
            labels[~np.isfinite(labels)] = -1
            source = "external"

        if labels is None and self._prev_semantic_labels is not None and self._prev_rgb is not None:
            propagated = self._propagate_semantics(self._prev_rgb, rgb, self._prev_semantic_labels, self._prev_semantic_scores)
            if propagated is not None:
                labels, scores = propagated
                source = "propagated"

        if labels is None:
            labels, scores = self._infer_semantics(image)
            source = "inferred"

        if labels is None:
            self._prev_rgb = None
            self._prev_semantic_labels = None
            self._prev_semantic_scores = None
            self._prev_semantic_source = None
            return None

        labels = labels.astype(np.int32)
        if scores is not None:
            scores = scores.astype(np.float32)

        self._prev_rgb = rgb.copy()
        self._prev_semantic_labels = labels.copy()
        self._prev_semantic_scores = scores.copy() if scores is not None else None
        self._prev_semantic_source = source

        return {"labels": labels, "scores": scores, "source": source or "unknown"}

    def _load_segmentation_model(self):
        if self.segmentation_model_type is None:
            return None
        if self._segmentation_model is not None:
            return self._segmentation_model

        try:
            import torchvision
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Semantic guidance requires torchvision to be installed.") from exc

        model_id = self.segmentation_model_type.lower()
        if model_id == "deeplabv3_mobilenet_v3_large":
            from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

            weights_enum = getattr(
                torchvision.models.segmentation,
                "DeepLabV3_MobileNet_V3_Large_Weights",
                None,
            )
            if weights_enum is not None:
                weights = weights_enum.DEFAULT
                model = deeplabv3_mobilenet_v3_large(weights=weights)
                mean = torch.tensor(weights.meta.get("mean", _DEFAULT_SEG_MEAN.tolist()), dtype=torch.float32)
                std = torch.tensor(weights.meta.get("std", _DEFAULT_SEG_STD.tolist()), dtype=torch.float32)
            else:  # pragma: no cover - legacy torchvision fallback
                model = deeplabv3_mobilenet_v3_large(pretrained=True)
                mean = torch.from_numpy(_DEFAULT_SEG_MEAN)
                std = torch.from_numpy(_DEFAULT_SEG_STD)
        else:  # pragma: no cover - defensive
            raise ValueError(f"Unsupported segmentation model '{self.segmentation_model_type}'.")

        model.to(self.device)
        model.eval()
        self._segmentation_model = model
        self._segmentation_norm_mean = mean.float()
        self._segmentation_norm_std = std.float()
        return self._segmentation_model

    def _infer_semantics(self, image: Image.Image) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        model = self._load_segmentation_model()
        if model is None:
            return None, None

        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0)
        tensor = tensor.to(self.device)

        if self._segmentation_norm_mean is not None:
            mean = self._segmentation_norm_mean
        else:
            mean = torch.from_numpy(_DEFAULT_SEG_MEAN).float()
        if self._segmentation_norm_std is not None:
            std = self._segmentation_norm_std
        else:
            std = torch.from_numpy(_DEFAULT_SEG_STD).float()
        mean = mean.to(tensor.device).view(1, -1, 1, 1)
        std = std.to(tensor.device).view(1, -1, 1, 1)
        tensor = (tensor - mean) / torch.clamp(std, min=1e-6)

        with torch.no_grad():
            output = model(tensor)["out"]
            probs = torch.softmax(output, dim=1)
            scores, labels = torch.max(probs, dim=1)

        labels_np = labels.squeeze(0).cpu().numpy().astype(np.int32)
        scores_np = scores.squeeze(0).cpu().numpy().astype(np.float32)

        low_confidence = scores_np < self.semantic_confidence_threshold
        labels_np[low_confidence] = -1
        scores_np[low_confidence] = 0.0
        return labels_np, scores_np

    def _propagate_semantics(
        self,
        prev_rgb: np.ndarray,
        curr_rgb: np.ndarray,
        prev_labels: np.ndarray,
        prev_scores: Optional[np.ndarray],
    ) -> Optional[Tuple[np.ndarray, Optional[np.ndarray]]]:
        try:
            import cv2
        except ImportError:  # pragma: no cover - optional dependency
            return None

        if prev_rgb.shape != curr_rgb.shape:
            return None
        if prev_labels.shape != prev_rgb.shape[:2]:
            return None

        prev_gray = cv2.cvtColor(np.clip(prev_rgb * 255.0, 0.0, 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        curr_gray = cv2.cvtColor(np.clip(curr_rgb * 255.0, 0.0, 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            curr_gray,
            prev_gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=21,
            iterations=3,
            poly_n=5,
            poly_sigma=1.1,
            flags=0,
        )

        h, w = prev_labels.shape
        grid_x, grid_y = np.meshgrid(
            np.arange(w, dtype=np.float32),
            np.arange(h, dtype=np.float32),
        )
        map_x = grid_x + flow[..., 0]
        map_y = grid_y + flow[..., 1]

        warped_labels = cv2.remap(
            prev_labels.astype(np.float32),
            map_x,
            map_y,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=-1,
        ).astype(np.int32)

        if prev_scores is not None:
            warped_scores = cv2.remap(
                prev_scores.astype(np.float32),
                map_x,
                map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0.0,
            ).astype(np.float32)
        else:
            warped_scores = None

        valid = warped_labels >= 0
        if not valid.any():
            return None

        valid_ratio = float(valid.mean())
        if valid_ratio < 0.4:
            return None

        if warped_scores is not None:
            warped_scores[~valid] = 0.0

        return warped_labels, warped_scores

    def _blend_semantics(
        self,
        depth: np.ndarray,
        masks: Sequence[np.ndarray],
        quantiles: np.ndarray,
        semantics: Dict[str, np.ndarray],
    ) -> List[np.ndarray]:
        labels = semantics.get("labels")
        if labels is None:
            return list(masks)
        if labels.shape != depth.shape:
            raise ValueError("Semantic label map shape mismatch.")

        scores = semantics.get("scores")

        assignments = np.full(depth.shape, -1, dtype=np.int32)
        for idx, mask in enumerate(masks):
            assignments[mask] = idx

        updated = assignments.copy()
        valid_labels = labels >= 0
        label_ids = np.unique(labels[valid_labels])
        for label_id in label_ids:
            label_mask = labels == label_id
            if not label_mask.any():  # pragma: no cover - defensive
                continue

            if scores is not None:
                mean_score = float(scores[label_mask].mean())
                if mean_score < self.semantic_confidence_threshold:
                    continue

            depth_values = depth[label_mask]
            depth_values = depth_values[np.isfinite(depth_values)]
            if depth_values.size == 0:
                continue

            depth_std = float(np.std(depth_values))
            if depth_std <= self.semantic_depth_std_threshold:
                median_depth = float(np.median(depth_values))
                layer_idx = self._quantile_index(median_depth, quantiles)
            else:
                current = updated[label_mask]
                current = current[current >= 0]
                if current.size == 0:
                    layer_idx = self._quantile_index(float(np.median(depth_values)), quantiles)
                else:
                    counts = np.bincount(current, minlength=len(masks))
                    layer_idx = int(np.argmax(counts))

            updated[label_mask] = layer_idx

        missing = updated < 0
        if missing.any():
            fallback = self._vectorized_quantile_index(depth[missing], quantiles)
            updated[missing] = fallback

        blended_masks = [(updated == i) for i in range(len(masks))]
        return blended_masks

    def _quantile_index(self, value: float, quantiles: np.ndarray) -> int:
        if not np.isfinite(value):
            return max(0, len(quantiles) - 2)
        idx = int(np.searchsorted(quantiles, value, side="right") - 1)
        idx = max(0, min(len(quantiles) - 2, idx))
        return idx

    def _vectorized_quantile_index(self, values: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        result = np.searchsorted(quantiles, values, side="right") - 1
        result = np.clip(result, 0, len(quantiles) - 2)
        result[~np.isfinite(values)] = len(quantiles) - 2
        return result.astype(np.int32)

    def _layer_semantic_stats(
        self, mask: np.ndarray, semantics: Optional[Dict[str, np.ndarray]]
    ) -> Dict[str, float]:
        if semantics is None:
            return {}
        labels = semantics.get("labels")
        if labels is None:
            return {}
        layer_labels = labels[mask]
        layer_labels = layer_labels[layer_labels >= 0]
        if layer_labels.size == 0:
            return {}
        values, counts = np.unique(layer_labels, return_counts=True)
        if values.size == 0:
            return {}
        top_index = int(np.argmax(counts))
        return {
            "semantic_top_label": int(values[top_index]),
            "semantic_top_ratio": float(counts[top_index] / layer_labels.size),
        }

    def _load_image(self, image: ImageLike) -> Tuple[Image.Image, Optional[Path]]:
        if isinstance(image, Image.Image):
            return image, None
        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                image = np.stack([image] * 3, axis=-1)
            if image.ndim != 3 or image.shape[2] not in (3, 4):
                raise ValueError("ndarray inputs must be H×W×3 or H×W×4 arrays.")
            return Image.fromarray(image.astype(np.uint8)), None
        path = Path(image)
        if not path.exists():
            raise FileNotFoundError(path)
        return Image.open(path), path

    def _build_alpha(self, mask: np.ndarray) -> np.ndarray:
        alpha = np.clip(mask, 0.0, 1.0).astype(np.float32)
        if self.edge_softening_px <= 0.0:
            return alpha

        kernel_size = int(max(3, round(self.edge_softening_px * 4) | 1))
        sigma = max(self.edge_softening_px, 1e-3)
        kernel = self._gaussian_kernel(kernel_size, sigma)

        mask_t = torch.from_numpy(alpha)[None, None, ...]
        blurred = F.conv2d(mask_t, kernel, padding=kernel_size // 2)
        blurred = torch.clamp(blurred, 0.0, 1.0)
        blurred = torch.maximum(mask_t, blurred)
        return blurred.squeeze(0).squeeze(0).numpy()

    def _gaussian_kernel(self, kernel_size: int, sigma: float) -> torch.Tensor:
        key = (kernel_size, float(sigma))
        cached = self._gaussian_kernel_cache.get(key)
        if cached is not None:
            return cached

        coords = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.0
        gauss = torch.exp(-(coords**2) / (2 * sigma**2))
        kernel_2d = gauss[:, None] @ gauss[None, :]
        kernel_2d /= kernel_2d.sum()
        kernel = kernel_2d[None, None, ...]
        self._gaussian_kernel_cache[key] = kernel
        return kernel

    def _progressive_inpaint(self, rgb: np.ndarray, hole_mask: np.ndarray) -> np.ndarray:
        if not hole_mask.any():
            return rgb

        device = self.device if self.device.type == "cuda" else torch.device("cpu")
        img = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)
        known = torch.from_numpy((~hole_mask).astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
        holes = 1.0 - known

        kernel = torch.ones((1, 1, 3, 3), device=device)
        kernel_rgb = kernel.repeat(img.shape[1], 1, 1, 1)

        for _ in range(self.occlusion_fill_iterations):
            neighbor_counts = F.conv2d(known, kernel, padding=1)
            neighbor_rgb = F.conv2d(img * known, kernel_rgb, padding=1, groups=img.shape[1])

            new_pixels = (neighbor_counts > 0) & (holes > 0)
            if not bool(new_pixels.any()):
                break

            fill_rgb = neighbor_rgb / torch.clamp(neighbor_counts, min=1.0)
            new_pixels_rgb = new_pixels.expand(-1, img.shape[1], -1, -1)
            img = torch.where(new_pixels_rgb, fill_rgb, img)
            known = torch.where(new_pixels, torch.ones_like(known), known)
            holes = 1.0 - known

        return img.squeeze(0).cpu().permute(1, 2, 0).numpy()

    def _smooth_depth(self, depth: np.ndarray) -> np.ndarray:
        """Apply an optional Gaussian smoothing to the depth map.

        The filter respects invalid pixels (NaNs) by normalising with the
        accumulated mask weights so that missing values do not bleed in.
        """

        if self.depth_smoothing_sigma <= 0.0:
            return depth

        sigma = self.depth_smoothing_sigma
        kernel_size = int(max(3, round(sigma * 6) | 1))
        kernel = self._gaussian_kernel(kernel_size, sigma)

        depth_cpu = depth.copy()
        finite_mask = np.isfinite(depth_cpu).astype(np.float32)
        depth_cpu = np.nan_to_num(depth_cpu, nan=0.0).astype(np.float32)

        depth_t = torch.from_numpy(depth_cpu)[None, None, ...]
        mask_t = torch.from_numpy(finite_mask)[None, None, ...]

        with torch.no_grad():
            filtered = F.conv2d(depth_t * mask_t, kernel, padding=kernel_size // 2)
            weights = F.conv2d(mask_t, kernel, padding=kernel_size // 2)
            filtered /= torch.clamp(weights, min=1e-6)

        result = filtered.squeeze(0).squeeze(0).numpy()
        result[finite_mask == 0.0] = np.nan
        return result

    def _refine_mask(self, mask: np.ndarray) -> np.ndarray:
        """Close small holes in the binary support mask via morphology."""

        if self.mask_morphology_radius <= 0:
            return mask

        radius = self.mask_morphology_radius
        kernel_size = radius * 2 + 1
        kernel = torch.ones((1, 1, kernel_size, kernel_size), dtype=torch.float32)

        mask_t = torch.from_numpy(mask.astype(np.float32))[None, None, ...]
        # Dilation followed by erosion = closing, fills tiny gaps along edges.
        dilated = (F.conv2d(mask_t, kernel, padding=radius) > 0).float()
        eroded = (F.conv2d(dilated, kernel, padding=radius) >= kernel.numel()).float()
        return eroded.squeeze(0).squeeze(0).numpy().astype(bool)

    def _load_midas(self):
        if self._midas_model is None or self._midas_transform is None:
            self._midas_model = torch.hub.load("intel-isl/MiDaS", self.depth_model_type)
            self._midas_model.to(self.device)
            self._midas_model.eval()

            transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            if self.depth_model_type in ("DPT_Large", "DPT_Hybrid"):
                self._midas_transform = transforms.dpt_transform
            else:
                self._midas_transform = transforms.small_transform
        return self._midas_model, self._midas_transform

    @staticmethod
    def _auto_device() -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
