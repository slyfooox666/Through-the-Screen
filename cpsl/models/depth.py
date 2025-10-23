"""Depth estimation wrappers for CPSL."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import sys

import cv2
import numpy as np
import torch
from torch import nn


@dataclass
class DepthEstimator:
    """Dispatcher that supports MiDaS and Depth Anything backends."""

    model_name: str = "DPT_Hybrid"
    device: str = "cuda"
    checkpoint_path: Optional[Path] = None
    repo_path: Optional[Path] = None
    input_size: int = 518

    def __post_init__(self) -> None:
        if isinstance(self.model_name, Path):
            self.model_name = str(self.model_name)
        if isinstance(self.checkpoint_path, str):
            self.checkpoint_path = Path(self.checkpoint_path)
        if isinstance(self.repo_path, str):
            self.repo_path = Path(self.repo_path)
        if self.checkpoint_path is None:
            potential = Path(self.model_name)
            if potential.suffix in {".pth", ".pt"} and potential.exists():
                self.checkpoint_path = potential
        if self.repo_path is not None:
            self.repo_path = self.repo_path.resolve()
        if self.checkpoint_path is not None:
            self.checkpoint_path = self.checkpoint_path.resolve()
        self._backend = self._create_backend()

    def _create_backend(self) -> "_BaseDepthBackend":
        model_key_raw = str(self.model_name)
        model_key = model_key_raw.lower()
        device = self._resolve_device()
        if self.checkpoint_path is not None or "depth_anything_v2" in model_key:
            return _DepthAnythingV2Backend(
                model_key=model_key,
                device=device,
                checkpoint_path=self.checkpoint_path,
                repo_path=self.repo_path,
                input_size=self.input_size,
            )
        if model_key.startswith("depth_anything"):
            return _DepthAnythingBackend(model_key, device)
        return _MiDaSBackend(self.model_name, device)

    def _resolve_device(self) -> torch.device:
        if not torch.cuda.is_available() or self.device == "cpu":
            return torch.device("cpu")
        return torch.device(self.device)

    def predict(self, frame_bgr: np.ndarray) -> np.ndarray:
        return self._backend.predict(frame_bgr)


class _BaseDepthBackend:
    """Base class for depth estimation backends."""

    def __init__(self, device: torch.device) -> None:
        self.device = device

    def _post_process(self, depth: torch.Tensor, image_size) -> np.ndarray:
        depth = depth.unsqueeze(1)
        depth = torch.nn.functional.interpolate(
            depth,
            size=image_size,
            mode="bicubic",
            align_corners=False,
        )
        depth = depth.squeeze(1)
        depth_np = depth.squeeze().cpu().numpy().astype(np.float32)
        depth_np -= depth_np.min()
        max_val = depth_np.max()
        if max_val > 0:
            depth_np /= max_val
        return depth_np

    def predict(self, frame_bgr: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class _MiDaSBackend(_BaseDepthBackend):
    """MiDaS depth inference via torch.hub."""

    def __init__(self, model_name: str, device: torch.device) -> None:
        super().__init__(device)
        self.model_name = model_name
        self._load_model()

    def _load_model(self) -> None:
        self.model: nn.Module = torch.hub.load("intel-isl/MiDaS", self.model_name)
        self.model.to(self.device)
        self.model.eval()
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if "DPT" in self.model_name:
            self.transform = transforms.dpt_transform
        else:
            self.transform = transforms.small_transform

    @torch.inference_mode()
    def predict(self, frame_bgr: np.ndarray) -> np.ndarray:
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("MiDaS backend expects a BGR colour frame.")

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(frame_rgb).to(self.device)
        prediction: torch.Tensor = self.model(input_batch)
        return self._post_process(prediction, frame_rgb.shape[:2])


class _DepthAnythingBackend(_BaseDepthBackend):
    """Depth Anything via Hugging Face transformers integration."""

    HF_REPOS: Dict[str, str] = {
        "depth_anything_vits14": "LiheYoung/depth-anything-small-hf",
        "depth_anything_vitb14": "LiheYoung/depth-anything-base-hf",
        "depth_anything_vitl14": "LiheYoung/depth-anything-large-hf",
        "depth_anything_l": "LiheYoung/depth-anything-large-hf",
        "depth_anything_b": "LiheYoung/depth-anything-base-hf",
        "depth_anything_s": "LiheYoung/depth-anything-small-hf",
    }

    def __init__(self, model_key: str, device: torch.device) -> None:
        super().__init__(device)
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required for Depth Anything. Install via `pip install transformers`."
            ) from exc

        repo_id = self.HF_REPOS.get(model_key, model_key)
        self.processor = AutoImageProcessor.from_pretrained(repo_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(repo_id)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def predict(self, frame_bgr: np.ndarray) -> np.ndarray:
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("Depth Anything backend expects a BGR colour frame.")

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=frame_rgb, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        prediction: torch.Tensor = outputs.predicted_depth
        return self._post_process(prediction, frame_rgb.shape[:2])


class _DepthAnythingV2Backend(_BaseDepthBackend):
    """Depth Anything V2 backend using local checkpoints."""

    MODEL_CONFIGS: Dict[str, Dict[str, object]] = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
    }

    def __init__(
        self,
        model_key: str,
        device: torch.device,
        checkpoint_path: Optional[Path],
        repo_path: Optional[Path],
        input_size: int,
    ) -> None:
        super().__init__(device)
        self.input_size = input_size
        self.encoder = self._determine_encoder(model_key)
        self.repo_dir = self._resolve_repo_dir(repo_path)
        if str(self.repo_dir) not in sys.path:
            sys.path.insert(0, str(self.repo_dir))
        try:
            from depth_anything_v2.dpt import DepthAnythingV2  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Failed to import Depth Anything V2 package. Set `preprocess.depth_repo` to the repository root."
            ) from exc

        cfg = self.MODEL_CONFIGS[self.encoder]
        self.model = DepthAnythingV2(**cfg)
        checkpoint = self._resolve_checkpoint_path(checkpoint_path)
        state_dict = torch.load(str(checkpoint), map_location="cpu")
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def _determine_encoder(self, model_key: str) -> str:
        for candidate in ("vits", "vitb", "vitl", "vitg"):
            if candidate in model_key:
                return candidate
        return "vitb"

    def _resolve_repo_dir(self, repo_path: Optional[Path]) -> Path:
        candidates = []
        if repo_path is not None:
            candidates.append(repo_path)
        default_dir = Path(__file__).resolve().parents[2] / "Depth-Anything-V2"
        candidates.append(default_dir)
        for candidate in candidates:
            if candidate is not None and candidate.exists():
                return candidate
        raise RuntimeError(
            "Depth Anything V2 repository not found. Provide `preprocess.depth_repo` pointing to the repo root."
        )

    def _resolve_checkpoint_path(self, checkpoint_path: Optional[Path]) -> Path:
        if checkpoint_path is not None and checkpoint_path.exists():
            return checkpoint_path
        fallback = self.repo_dir / "checkpoints" / f"depth_anything_v2_{self.encoder}.pth"
        if fallback.exists():
            return fallback
        raise FileNotFoundError(
            f"Depth Anything V2 checkpoint not found. Expected at {fallback} or provide `preprocess.depth_checkpoint`."
        )

    @torch.inference_mode()
    def predict(self, frame_bgr: np.ndarray) -> np.ndarray:
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("Depth Anything V2 backend expects a BGR colour frame.")

        depth = self.model.infer_image(frame_bgr, input_size=self.input_size)
        depth = depth.astype(np.float32)
        depth -= depth.min()
        max_val = depth.max()
        if max_val > 0:
            depth /= max_val
        return depth
