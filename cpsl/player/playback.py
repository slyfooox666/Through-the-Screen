"""Geometry-aware CPSL playback module."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from cpsl.config import CPSLConfig, ViewerIntrinsics
from cpsl.player import synth
from cpsl.utils.video import VideoWriter


@dataclass
class LoadedLayer:
    index: int
    class_name: str
    depth_mean: float
    depth_std: float
    color_srgb: np.ndarray  # premultiplied RGB float32 range [0,1]
    alpha: np.ndarray  # float32 range [0,1]
    depth_map: np.ndarray  # float32 range [0,1]
    normal: Optional[np.ndarray] = None


@dataclass
class CameraPose:
    rotation: np.ndarray  # 3x3
    translation: np.ndarray  # 3,


class CPSLPlayer:
    """Produce novel-view frames from CPSL layers using plane-induced homographies."""

    def __init__(self, config: CPSLConfig) -> None:
        self.config = config
        self.output_root = config.io.output_root
        self.metadata_path = self.output_root / "metadata.json"
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {self.metadata_path}")
        self.metadata = self._load_metadata()
        self.camera_trace = self._load_trace(config.playback.trace_path)
        self.random_seed = config.playback.random_seed
        self._cached_random_walk: Optional[List[np.ndarray]] = None

        self.target_intrinsics = self._compute_target_intrinsics(config.playback.viewer_intrinsics)
        self.source_intrinsics = self._compute_source_intrinsics()

    def _load_metadata(self) -> Dict[str, object]:
        with self.metadata_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def render_to_video(self, output_path: Path) -> None:
        video_meta = self.metadata["video"]
        width = int(video_meta["width"])
        height = int(video_meta["height"])
        fps = float(self.config.playback.fps or video_meta.get("fps", 30))

        frames_meta = self.metadata["frames"]
        total_frames = len(frames_meta)
        poses = self._compute_camera_poses(total_frames)

        with VideoWriter(output_path, fps=fps, frame_size=(width, height)) as writer:
            for idx, frame in tqdm(enumerate(frames_meta), total=total_frames, desc="CPSL playback"):
                layers = self._load_layers_for_frame(frame)
                rendered = self._render_geometry(layers, poses[idx], (width, height))
                writer.write_frames([rendered])

    def _compute_camera_poses(self, total_frames: int) -> List[CameraPose]:
        if total_frames <= 0:
            return []

        translations: List[np.ndarray]
        rotations: List[np.ndarray]
        if self.camera_trace is not None:
            translations = []
            rotations = []
            last_translation = np.zeros(3, dtype=np.float32)
            last_rotation = np.eye(3, dtype=np.float32)
            for frame_idx in range(total_frames):
                if frame_idx in self.camera_trace:
                    last_translation, last_rotation = self.camera_trace[frame_idx]
                translations.append(last_translation.copy())
                rotations.append(last_rotation.copy())
        else:
            translations, rotations = self._generate_random_walk(total_frames)

        return [CameraPose(rotation=rot, translation=trans) for rot, trans in zip(rotations, translations)]

    def _generate_random_walk(self, total_frames: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        if self._cached_random_walk is not None and len(self._cached_random_walk) == total_frames:
            translations = [vec.copy() for vec in self._cached_random_walk]
        else:
            rng = np.random.default_rng(self.random_seed)
            translations = []
            prev = np.zeros(3, dtype=np.float32)
            step_scale = 0.02  # metres per frame
            smoothing = 0.92  # closer to 1 => smoother drift
            max_offset = self.config.playback.max_view_offset
            for _ in range(total_frames):
                step = rng.normal(loc=0.0, scale=step_scale, size=3)
                prev = smoothing * prev + (1.0 - smoothing) * step
                prev = np.clip(prev, -max_offset, max_offset)
                translations.append(prev.copy())
            self._cached_random_walk = translations

        rotations = [np.eye(3, dtype=np.float32) for _ in range(total_frames)]
        return translations, rotations

    def _load_trace(self, trace_path: Optional[Path]) -> Optional[Dict[int, Tuple[np.ndarray, np.ndarray]]]:
        if trace_path is None:
            return None
        if not trace_path.exists():
            raise FileNotFoundError(f"Trace file not found: {trace_path}")
        trace: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        with trace_path.open("r", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            for row in reader:
                if not row:
                    continue
                first = row[0].strip()
                if not first or first.startswith("#"):
                    continue
                token = first.lstrip("-")
                if not token.isdigit():
                    # Allow header rows
                    continue
                if len(row) < 3:
                    raise ValueError(f"Trace row must contain at least 3 values: {row}")
                try:
                    frame_id = int(first)
                    tx = float(row[1])
                    ty = float(row[2])
                    tz = float(row[3]) if len(row) > 3 else 0.0
                    yaw = float(row[4]) if len(row) > 4 else 0.0
                    pitch = float(row[5]) if len(row) > 5 else 0.0
                    roll = float(row[6]) if len(row) > 6 else 0.0
                except ValueError as exc:
                    raise ValueError(f"Invalid numeric values in trace row: {row}") from exc
                translation = np.array([tx, ty, tz], dtype=np.float32)
                rotation = self._euler_to_matrix(yaw, pitch, roll)
                trace[frame_id] = (translation, rotation)
        return trace

    def _load_layers_for_frame(self, frame_meta: Dict[str, object]) -> List[LoadedLayer]:
        layers: List[LoadedLayer] = []
        for layer_meta in sorted(frame_meta["layers"], key=lambda l: l["index"]):
            rgba_path = self.output_root / layer_meta["rgba_path"]
            depth_path = self.output_root / layer_meta["depth_path"]
            if not rgba_path.exists():
                raise FileNotFoundError(rgba_path)
            rgba_bgra = cv2.imread(str(rgba_path), cv2.IMREAD_UNCHANGED)
            if rgba_bgra is None or rgba_bgra.shape[2] != 4:
                raise RuntimeError(f"Failed to load RGBA layer: {rgba_path}")
            bgr = rgba_bgra[:, :, :3].astype(np.float32) / 255.0
            alpha = rgba_bgra[:, :, 3].astype(np.float32) / 255.0
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            depth_map = np.load(depth_path).astype(np.float32)
            layers.append(
                LoadedLayer(
                    index=int(layer_meta["index"]),
                    class_name=layer_meta["class_name"],
                    depth_mean=float(layer_meta["depth_mean"]),
                    depth_std=float(layer_meta["depth_std"]),
                    color_srgb=rgb,
                    alpha=alpha,
                    depth_map=depth_map,
                    normal=None,
                )
            )
        return layers

    def _render_geometry(
        self,
        layers: Sequence[LoadedLayer],
        pose: CameraPose,
        output_size: Tuple[int, int],
    ) -> np.ndarray:
        payload: List[Tuple[np.ndarray, np.ndarray, np.ndarray, float, Optional[np.ndarray]]] = []
        for layer in layers:
            disparity = np.clip(layer.depth_map, 1e-6, None)
            metric_depth_map = 1.0 / disparity  # larger disparity (near) => smaller metric depth
            if np.any(layer.alpha > 1e-3):
                plane_depth = float(np.median(metric_depth_map[layer.alpha > 1e-3]))
            else:
                plane_depth = float(np.median(metric_depth_map))
            payload.append(
                (
                    layer.color_srgb,
                    layer.alpha,
                    metric_depth_map,
                    plane_depth,
                    layer.normal,
                )
            )
        payload.sort(key=lambda item: item[3])
        return synth.render_frame(
            layers=payload,
            Ks=self.source_intrinsics,
            Kt=self.target_intrinsics,
            R=pose.rotation,
            t=pose.translation,
            output_size=output_size,
            edge_smooth_px=2,
            use_zbuffer=True,
        )

    def _compute_source_intrinsics(self) -> np.ndarray:
        video_meta = self.metadata["video"]
        intrinsics_meta = video_meta.get("intrinsics", {})
        fx = intrinsics_meta.get("fx")
        fy = intrinsics_meta.get("fy")
        cx = intrinsics_meta.get("cx")
        cy = intrinsics_meta.get("cy")

        if None not in (fx, fy, cx, cy):
            return synth.build_intrinsic_matrix(float(fx), float(fy), float(cx), float(cy))

        # Fallback: assume source intrinsics match the target intrinsics (identity when R=I, t=0).
        return self.target_intrinsics.copy()

    def _compute_target_intrinsics(self, intr: ViewerIntrinsics) -> np.ndarray:
        return synth.build_intrinsic_matrix(intr.fx, intr.fy, intr.cx, intr.cy)

    def _euler_to_matrix(self, yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
        yaw = np.deg2rad(yaw_deg)
        pitch = np.deg2rad(pitch_deg)
        roll = np.deg2rad(roll_deg)

        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cr, sr = np.cos(roll), np.sin(roll)

        R_yaw = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32)
        R_pitch = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], dtype=np.float32)
        R_roll = np.array([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        return (R_roll @ R_pitch @ R_yaw).astype(np.float32)
