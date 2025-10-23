"""Offline preprocessing pipeline for CPSL prototype."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from tqdm import tqdm

from cpsl.config import CPSLConfig
from cpsl.models.depth import DepthEstimator
from cpsl.pipeline.depth_regions import cluster_depth_regions, smooth_depth
from cpsl.pipeline.layers import LayerData, generate_layers
from cpsl.utils.fs import ensure_dir, save_json
from cpsl.utils.video import VideoReader


@dataclass
class FrameLayerMetadata:
    index: int
    class_name: str
    depth_mean: float
    depth_std: float
    rgba_path: str
    depth_path: str


@dataclass
class FrameMetadata:
    index: int
    layers: List[FrameLayerMetadata]


class CPSLPreprocessor:
    """Runs the offline CPSL preprocessing to produce layers and metadata."""

    def __init__(self, config: CPSLConfig) -> None:
        self.config = config
        self._depth = DepthEstimator(
            model_name=config.preprocess.depth_model,
            device=config.preprocess.device,
            checkpoint_path=config.preprocess.depth_checkpoint,
            repo_path=config.preprocess.depth_repo,
            input_size=config.preprocess.depth_input_size,
        )

    def run(self) -> Dict[str, object]:
        io_cfg = self.config.io
        ensure_dir(io_cfg.output_root)

        metadata: Dict[str, object] = {
            "video": {},
            "frames": [],
        }

        with VideoReader(io_cfg.input_video) as reader:
            width, height, fps = reader.metadata
            metadata["video"] = {"width": width, "height": height, "fps": fps}
            for frame_idx, frame in tqdm(reader.frames(), desc="CPSL preprocess"):
                frame_dir = io_cfg.output_root / f"frame_{frame_idx:04d}"
                ensure_dir(frame_dir)
                layers = self._process_frame(
                    frame_idx=frame_idx,
                    frame_bgr=frame,
                    frame_dir=frame_dir,
                )
                metadata["frames"].append(
                    asdict(
                        FrameMetadata(
                            index=frame_idx,
                            layers=[
                                FrameLayerMetadata(
                                    index=layer.index,
                                    class_name=layer.class_name,
                                    depth_mean=layer.depth_mean,
                                    depth_std=layer.depth_std,
                                    rgba_path=str((frame_dir / f"layer_{layer.index:02d}.png").relative_to(io_cfg.output_root)),
                                    depth_path=str((frame_dir / f"layer_{layer.index:02d}_depth.npy").relative_to(io_cfg.output_root)),
                                )
                                for layer in layers
                            ],
                        )
                    )
                )

        save_json(metadata, io_cfg.output_root / "metadata.json")
        return metadata

    def _process_frame(
        self,
        frame_idx: int,
        frame_bgr: np.ndarray,
        frame_dir: Path,
    ) -> List[LayerData]:
        depth_map = self._depth.predict(frame_bgr)
        depth_for_regions = smooth_depth(depth_map, self.config.preprocess.depth_smooth_kernel)
        labels = cluster_depth_regions(
            depth_map=depth_for_regions,
            target_layers=self.config.preprocess.target_layers,
            spatial_weight=self.config.preprocess.spatial_weight,
            min_area_ratio=self.config.preprocess.min_area_ratio,
        )
        class_names = [f"depth_layer_{idx}" for idx in range(labels.max() + 1)]
        layers = generate_layers(
            frame_bgr=frame_bgr,
            depth_map=depth_map,
            labels=labels,
            class_names=class_names,
            target_layers=self.config.preprocess.target_layers,
            promote_classes=self.config.preprocess.promote_classes,
            boundary_band_px=self.config.preprocess.boundary_band_px,
            soft_alpha_sigma=self.config.preprocess.soft_alpha_sigma,
            soft_alpha_max=self.config.preprocess.soft_alpha_max,
        )

        for layer in layers:
            rgba_path = frame_dir / f"layer_{layer.index:02d}.png"
            depth_path = frame_dir / f"layer_{layer.index:02d}_depth.npy"

            rgba_uint8 = np.clip(layer.rgba * 255.0, 0, 255).astype(np.uint8)
            bgr = rgba_uint8[:, :, :3]
            alpha = rgba_uint8[:, :, 3]
            bgra = np.dstack((bgr, alpha))
            cv2.imwrite(str(rgba_path), bgra)
            np.save(depth_path, layer.depth_map.astype(np.float16))

        return layers
