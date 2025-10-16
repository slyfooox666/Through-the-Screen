"""SAM-powered CPSL generator.

This module adds an alternative generator that relies on Segment Anything (SAM)
models—ideally SAM 2.0 running on a GPU server—for semantic-aware layer
construction.  The class mirrors :class:`MultiLayerGenerator` but swaps the
semantic segmentation backend with SAM masks.

Two execution modes are supported:

``sam_mode="local"``
    Loads a SAM backbone in-process (requires the ``sam2`` package for SAM 2.0 or
    ``segment-anything`` for SAM 1.x).  Use this mode on remote GPU servers.

``sam_mode="http"``
    Sends frames to a remote inference service.  The service should accept a JSON
    payload containing a base64-encoded RGB image and return an array of masks in
    the format ``[{"score": float, "mask": <base64 PNG or RLE>}, ...]``.  This
    keeps the client lightweight if SAM runs elsewhere.

The resulting semantic label map is blended with the depth-based stratification
implemented in :class:`volumetric_layers.generator.MultiLayerGenerator`.
"""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

import torch

from .generator import Layer, LayeredFrameSet, MultiLayerGenerator

try:  # Optional local SAM 2.0 backend
    from sam2.build_sam2 import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from sam2.sam2_automatic_mask_generator import SAM2AutomaticMaskGenerator

    _HAS_SAM2 = True
except Exception:  # pragma: no cover - optional dependency
    SAM2AutomaticMaskGenerator = None
    SAM2ImagePredictor = None
    build_sam2 = None
    _HAS_SAM2 = False

try:  # Fallback to SAM v1 if available
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    _HAS_SAM = True
except Exception:  # pragma: no cover
    SamAutomaticMaskGenerator = None
    sam_model_registry = None
    _HAS_SAM = False

try:  # Optional HTTP client
    import requests
except Exception:  # pragma: no cover
    requests = None


class SAMMultiLayerGenerator(MultiLayerGenerator):
    """Drop-in replacement that sources semantics from SAM masks.

    Parameters
    ----------
    sam_checkpoint : str
        Path to the SAM checkpoint (SAM 2.0 or SAM 1.x).
    sam_model : str, default ``"sam2_hiera_t"``
        Model identifier.  ``sam2_*`` expects the SAM 2.0 codebase, while other
        values fall back to SAM 1.x registries.
    sam_mode : {"local", "http"}
        How masks are produced.  ``local`` loads weights in-process; ``http``
        forwards frames to a remote endpoint.
    sam_endpoint : str, optional
        HTTP endpoint used when ``sam_mode='http'``.
    sam_headers : dict, optional
        Extra headers (e.g., API keys) for the HTTP call.
    max_sam_masks : int, default 5
        Only the top-N SAM masks (by score * area) are kept when forming the
        semantic label map.
    """

    def __init__(
        self,
        sam_checkpoint: Union[str, Path],
        sam_model: str = "sam2_hiera_t",
        sam_mode: str = "local",
        sam_endpoint: Optional[str] = None,
        sam_headers: Optional[Dict[str, str]] = None,
        max_sam_masks: int = 5,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ) -> None:
        kwargs.setdefault("segmentation_model_type", None)  # disable built-in semantics
        super().__init__(device=device, **kwargs)

        self.sam_checkpoint = Path(sam_checkpoint)
        if sam_mode == "local" and not self.sam_checkpoint.exists():
            raise FileNotFoundError(f"SAM checkpoint not found: {self.sam_checkpoint}")

        self.sam_model = sam_model
        self.sam_mode = sam_mode.lower()
        if self.sam_mode not in {"local", "http"}:
            raise ValueError("sam_mode must be 'local' or 'http'")
        self.sam_endpoint = sam_endpoint
        self.sam_headers = sam_headers or {}
        self.max_sam_masks = max(1, max_sam_masks)
        self._mask_generator = None
        self._sam_device = self.device if device is None else torch.device(device)

        if self.sam_mode == "http" and not self.sam_endpoint:
            raise ValueError("HTTP mode requires --sam-endpoint")

    # ------------------------------------------------------------------
    def generate(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        depth_map: Optional[np.ndarray] = None,
        num_layers: Optional[int] = None,
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
        depth_map = self._smooth_depth(depth_map)

        semantics = self._sam_semantics(rgb)

        target_layers = num_layers or self.num_layers
        masks, quantiles = self._stratify_depth(depth_map, target_layers)
        if semantics is not None:
            masks = self._blend_semantics(depth_map, masks, quantiles, semantics)
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
            stats.update(self._layer_semantic_stats(mask, semantics))
            layer = Layer(
                name=spec["name"],
                rgba=rgba,
                depth=depth_snapshot,
                mask=mask,
                statistics=stats,
            )
            layers.insert(0, layer)
            occlusion |= mask

        depth_meta = {
            "quantile_bounds": [float(b) for b in quantiles],
            "normalisation": self.depth_normalisation,
            "model": self.depth_model_type,
        }
        if semantics is not None:
            depth_meta["semantics"] = {
                "source": semantics.get("source"),
                "num_masks": int(semantics.get("num_masks", 0)),
            }

        return LayeredFrameSet(
            source_image_path=image_path,
            layers=layers,
            depth_metadata=depth_meta,
        )

    # ------------------------------------------------------------------
    def _sam_semantics(self, rgb: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
        if self.sam_mode == "http":
            return self._sam_http(rgb)
        else:
            return self._sam_local(rgb)

    def _sam_local(self, rgb: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
        generator = self._load_local_mask_generator()
        if generator is None:
            raise RuntimeError("SAM local mode requested but sam2/segment-anything not installed")

        image_uint8 = np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)
        masks = generator.generate(image_uint8)
        if not masks:
            return None

        sorted_masks = sorted(
            masks,
            key=lambda item: item.get("score", 0.0) * float(np.count_nonzero(item.get("segmentation", np.zeros_like(rgb[..., 0], dtype=bool)))),
            reverse=True,
        )[: self.max_sam_masks]

        h, w = rgb.shape[:2]
        labels = np.full((h, w), -1, dtype=np.int32)
        scores = np.zeros((h, w), dtype=np.float32)
        for label_idx, entry in enumerate(sorted_masks, start=1):
            seg = entry.get("segmentation")
            if seg is None:
                continue
            seg_bool = np.asarray(seg, dtype=bool)
            if seg_bool.shape != labels.shape:
                continue
            labels[seg_bool] = label_idx
            scores[seg_bool] = float(entry.get("score", 0.0))

        return {
            "labels": labels,
            "scores": scores,
            "source": self.sam_model,
            "num_masks": len(sorted_masks),
        }

    def _sam_http(self, rgb: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
        if requests is None:
            raise RuntimeError("requests is required for HTTP SAM mode")
        if not self.sam_endpoint:
            raise RuntimeError("sam_endpoint must be provided for HTTP SAM mode")

        image_uint8 = np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)
        buffer = io.BytesIO()
        Image.fromarray(image_uint8, mode="RGB").save(buffer, format="PNG")
        payload = {
            "image": base64.b64encode(buffer.getvalue()).decode("utf-8"),
            "max_masks": self.max_sam_masks,
        }
        response = requests.post(self.sam_endpoint, headers=self.sam_headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        masks = data.get("masks", [])
        if not masks:
            return None

        h, w = rgb.shape[:2]
        labels = np.full((h, w), -1, dtype=np.int32)
        scores = np.zeros((h, w), dtype=np.float32)
        for idx, entry in enumerate(masks[: self.max_sam_masks], start=1):
            mask_b64 = entry.get("mask")
            if not mask_b64:
                continue
            mask_bytes = base64.b64decode(mask_b64)
            mask_img = Image.open(io.BytesIO(mask_bytes)).convert("L")
            mask = np.asarray(mask_img, dtype=np.uint8) > 127
            labels[mask] = idx
            scores[mask] = float(entry.get("score", 0.0))

        return {
            "labels": labels,
            "scores": scores,
            "source": data.get("model", "sam-http"),
            "num_masks": len(masks),
        }

    def _load_local_mask_generator(self):
        if self._mask_generator is not None:
            return self._mask_generator

        model_name = self.sam_model.lower()
        if model_name.startswith("sam2"):
            if not _HAS_SAM2:
                raise RuntimeError(
                    "sam2 package not found. Install SAM 2.0 from https://github.com/facebookresearch/segment-anything-2"
                )
            sam = build_sam2(model_name, checkpoint=str(self.sam_checkpoint))
            sam.to(self._sam_device)
            generator = SAM2AutomaticMaskGenerator(sam)
        else:
            if not _HAS_SAM:
                raise RuntimeError(
                    "segment-anything package not found. Install with `pip install segment-anything`"
                )
            if model_name not in sam_model_registry:
                raise ValueError(f"Unknown SAM model '{self.sam_model}'.")
            sam = sam_model_registry[model_name](checkpoint=str(self.sam_checkpoint))
            sam.to(self._sam_device)
            generator = SamAutomaticMaskGenerator(sam)

        self._mask_generator = generator
        return self._mask_generator
