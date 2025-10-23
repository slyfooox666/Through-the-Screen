"""Semantic segmentation wrapper using torchvision models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import cv2
import numpy as np
import torch
from torchvision import transforms
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, deeplabv3_resnet50


@dataclass
class SemanticSegmenter:
    """Runs semantic segmentation and returns class indices per pixel."""

    device: str = "cuda"
    score_threshold: float = 0.3

    def __post_init__(self) -> None:
        if not torch.cuda.is_available() or self.device == "cpu":
            self.device = "cpu"
        self._device = torch.device(self.device)
        weights = DeepLabV3_ResNet50_Weights.DEFAULT
        self._model = deeplabv3_resnet50(weights=weights)
        self._model.to(self._device)
        self._model.eval()
        self.class_names = weights.meta.get("categories", [])
        self._preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @torch.inference_mode()
    def predict(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return label map and confidence."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor = self._preprocess(frame_rgb).unsqueeze(0).to(self._device)
        output = self._model(tensor)["out"]
        probs = torch.softmax(output, dim=1)
        confidence, labels = torch.max(probs, dim=1)
        return labels.squeeze(0).cpu().numpy(), confidence.squeeze(0).cpu().numpy()

