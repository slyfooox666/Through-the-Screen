"""Client-side compositor for gaze-adaptive rendering of layered frames."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from .generator import Layer, LayeredFrameSet


@dataclass
class GazeState:
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    zoom: float = 1.0  # 1.0 = original distance

    @property
    def yaw_rad(self) -> float:
        return math.radians(self.yaw_deg)

    @property
    def pitch_rad(self) -> float:
        return math.radians(self.pitch_deg)


class LayeredFrameComposer:
    def __init__(
        self,
        viewport_size: Optional[Tuple[int, int]] = None,
        near_plane: float = 0.1,
        far_plane: float = 10.0,
        fov_deg: float = 60.0,
        gaze_smoothing: float = 0.6,
        warp_supersample: int = 2,
        temporal_smoothing: float = 0.3,
        layer_scale_strength: float = 0.05,
        layer_min_scale: float = 1.0,
        layer_parallax_min: float = 0.1,
    ) -> None:
        self.viewport_size = viewport_size
        self.near_plane = near_plane
        self.far_plane = far_plane
        self.fov_deg = fov_deg
        self.gaze_smoothing = max(0.0, min(gaze_smoothing, 0.99))
        self.warp_supersample = max(1, warp_supersample)
        self.temporal_smoothing = max(0.0, min(temporal_smoothing, 0.99))
        self.layer_scale_strength = max(0.0, layer_scale_strength)
        self.layer_min_scale = max(0.0, min(layer_min_scale, 1.0))
        self.layer_parallax_min = max(0.0, min(layer_parallax_min, 1.0))
        self._prev_gaze: Optional[GazeState] = None
        self._prev_canvas: Optional[np.ndarray] = None

    def compose_from_directory(self, directory: Path | str, gaze: GazeState) -> Image.Image:
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(directory)

        metadata_path = directory / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError("Expected metadata.json in the layer directory")

        import json

        metadata = json.loads(metadata_path.read_text())
        gaze = self._smooth_gaze(gaze)

        layers_meta = metadata.get("layers") or []
        layer_dirs = [path for path in sorted(directory.iterdir()) if path.is_dir()]

        if not layers_meta:
            layers_meta = [{"name": path.name} for path in layer_dirs]
        else:
            unnamed = [info for info in layers_meta if not info.get("name")]
            if unnamed:
                if len(layer_dirs) < len(unnamed):
                    raise ValueError(
                        "Metadata is missing layer names and cannot be inferred from the directory."
                    )
                for info, layer_dir in zip(unnamed, layer_dirs):
                    info["name"] = layer_dir.name

        rgba_layers: List[np.ndarray] = []
        depth_layers: List[np.ndarray] = []
        for layer_info in layers_meta:
            name = layer_info.get("name")
            if name is None:
                raise ValueError("Each layer entry must have a name")
            layer_dir = directory / name
            rgba_path = next(layer_dir.glob("*_rgba.png"), None)
            depth_path = layer_dir / "depth.npy"
            if rgba_path is None or not depth_path.exists():
                raise FileNotFoundError(f"Missing assets for layer '{name}'")

            rgba_layers.append(np.asarray(Image.open(rgba_path).convert("RGBA"), dtype=np.float32) / 255.0)
            depth_layers.append(np.load(depth_path))

        return self.compose_layers(rgba_layers, depth_layers, gaze)

    def compose_from_frame(self, frame_dir: Path | str, gaze: GazeState) -> Image.Image:
        """Compose a frame stored in the canonical CPSL directory layout."""

        frame_path = Path(frame_dir)
        metadata_path = frame_path / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata.json in {frame_path}")

        layers_root = frame_path / "layers"
        if not layers_root.exists():
            raise FileNotFoundError(f"Missing 'layers' directory in {frame_path}")

        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)

        layer_names = [entry.get("name") for entry in metadata.get("layers", []) if entry.get("name")]
        if not layer_names:
            layer_names = sorted([p.name for p in layers_root.iterdir() if p.is_dir()])

        rgba_layers: List[np.ndarray] = []
        depth_layers: List[np.ndarray] = []
        for name in layer_names:
            layer_dir = layers_root / name
            rgba_path = layer_dir / "rgba.png"
            depth_npy = layer_dir / "depth.npy"
            if not rgba_path.exists():
                raise FileNotFoundError(f"Missing {rgba_path}")
            if depth_npy.exists():
                depth = np.load(depth_npy)
            else:
                depth_png = layer_dir / "depth.png"
                if not depth_png.exists():
                    raise FileNotFoundError(f"Missing depth map for layer {name}")
                depth = np.asarray(Image.open(depth_png), dtype=np.float32) / 65535.0
            rgba = np.asarray(Image.open(rgba_path).convert("RGBA"), dtype=np.float32) / 255.0
            rgba_layers.append(rgba)
            depth_layers.append(depth)

        return self.compose_layers(rgba_layers, depth_layers, gaze)

    def compose_layers(
        self,
        rgba_layers: Sequence[np.ndarray],
        depth_layers: Sequence[np.ndarray],
        gaze: GazeState,
    ) -> Image.Image:
        if len(rgba_layers) != len(depth_layers):
            raise ValueError("Depth and RGBA layer counts must match")
        if not rgba_layers:
            raise ValueError("At least one layer is required")

        height, width = rgba_layers[0].shape[:2]
        viewport_w, viewport_h = self.viewport_size or (width, height)

        projection = self._perspective_matrix(viewport_w, viewport_h)

        warped_layers: List[np.ndarray] = []
        total_layers = len(rgba_layers)
        for order, (rgba, depth) in enumerate(zip(rgba_layers, depth_layers)):
            if not np.isfinite(depth).any():
                continue
            median_depth = float(np.nanmedian(depth))
            progress = order / max(total_layers - 1, 1)
            base_scale = self.layer_min_scale + (1.0 - self.layer_min_scale) * progress
            scale = base_scale * (1.0 + self.layer_scale_strength * progress)
            parallax_mix = self.layer_parallax_min + (1.0 - self.layer_parallax_min) * progress
            layer_gaze = GazeState(
                yaw_deg=gaze.yaw_deg * parallax_mix,
                pitch_deg=gaze.pitch_deg * parallax_mix,
                zoom=1.0 + (gaze.zoom - 1.0) * parallax_mix,
            )
            view = self._view_matrix(layer_gaze)
            vertices = self._quad_vertices(width, height, median_depth, scale)
            warped = self._warp_layer(rgba, vertices, view, projection, (viewport_h, viewport_w))
            warped_layers.append(warped)

        if not warped_layers:
            raise ValueError("All layers were empty after warping")

        canvas = np.zeros((viewport_h, viewport_w, 4), dtype=np.float32)
        for layer in warped_layers:
            canvas = self._alpha_composite(canvas, layer)

        canvas = self._apply_temporal_smoothing(canvas)
        canvas_uint8 = np.clip(canvas * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(canvas_uint8, mode="RGBA")

    def render(
        self,
        layered_frame: LayeredFrameSet | Sequence[Layer],
        yaw_deg: float = 0.0,
        pitch_deg: float = 0.0,
        zoom: float = 1.0,
    ) -> np.ndarray:
        """Render a novel view from in-memory layers."""

        if isinstance(layered_frame, LayeredFrameSet):
            layers = layered_frame.layers
        else:
            layers = list(layered_frame)

        rgba_layers = [layer.rgba for layer in layers]
        depth_layers = [layer.depth for layer in layers]
        image = self.compose_layers(
            rgba_layers,
            depth_layers,
            GazeState(yaw_deg=yaw_deg, pitch_deg=pitch_deg, zoom=zoom),
        )
        return np.asarray(image.convert("RGB"))

    # ------------------------------------------------------------------
    def _perspective_matrix(self, viewport_w: int, viewport_h: int) -> np.ndarray:
        aspect = viewport_w / max(viewport_h, 1)
        f = 1.0 / math.tan(math.radians(self.fov_deg) / 2.0)
        near = self.near_plane
        far = self.far_plane
        return np.array(
            [
                [f / aspect, 0.0, 0.0, 0.0],
                [0.0, f, 0.0, 0.0],
                [0.0, 0.0, (far + near) / (near - far), (2 * far * near) / (near - far)],
                [0.0, 0.0, -1.0, 0.0],
            ],
            dtype=np.float32,
        )

    def _view_matrix(self, gaze: GazeState) -> np.ndarray:
        cy, sy = math.cos(gaze.yaw_rad), math.sin(gaze.yaw_rad)
        cp, sp = math.cos(gaze.pitch_rad), math.sin(gaze.pitch_rad)

        rot_yaw = np.array(
            [[cy, 0.0, sy, 0.0], [0.0, 1.0, 0.0, 0.0], [-sy, 0.0, cy, 0.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        rot_pitch = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, cp, -sp, 0.0], [0.0, sp, cp, 0.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        translation = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, -gaze.zoom], [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        return translation @ rot_pitch @ rot_yaw

    def _quad_vertices(self, width: int, height: int, depth: float, scale: float = 1.0) -> np.ndarray:
        x_min, x_max = -0.5, 0.5
        y_extent = 0.5 * height / max(width, 1)
        y_min, y_max = -y_extent, y_extent
        x_min *= scale
        x_max *= scale
        y_min *= scale
        y_max *= scale
        z = depth * 4.0 + 1.5
        return np.array(
            [
                [x_min, y_max, z, 1.0],
                [x_max, y_max, z, 1.0],
                [x_max, y_min, z, 1.0],
                [x_min, y_min, z, 1.0],
            ],
            dtype=np.float32,
        )

    def _warp_layer(
        self,
        rgba: np.ndarray,
        vertices: np.ndarray,
        view: np.ndarray,
        projection: np.ndarray,
        target_size: Tuple[int, int],
    ) -> np.ndarray:
        h, w = target_size
        factor = self.warp_supersample
        # Optionally oversample the projection grid to reduce aliasing before
        # average pooling back to the requested viewport resolution.
        sample_h, sample_w = (h * factor, w * factor) if factor > 1 else target_size

        clip = (projection @ view @ vertices.T).T
        ndc = clip[:, :3] / clip[:, 3:4]

        xs = ((ndc[:, 0] + 1.0) * 0.5) * (sample_w - 1)
        ys = ((ndc[:, 1] + 1.0) * 0.5) * (sample_h - 1)

        src_quad = np.array(
            [[0, 0], [rgba.shape[1] - 1, 0], [rgba.shape[1] - 1, rgba.shape[0] - 1], [0, rgba.shape[0] - 1]],
            dtype=np.float32,
        )
        dst_quad = np.stack([xs, ys], axis=1)

        H = self._homography(src_quad, dst_quad)
        warped = self._apply_homography(rgba, H, (sample_h, sample_w))
        if factor > 1:
            warped = self._downsample(warped, factor)
        return warped

    def _homography(self, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        A = []
        for (x, y), (X, Y) in zip(src, dst):
            A.append([-x, -y, -1, 0, 0, 0, x * X, y * X, X])
            A.append([0, 0, 0, -x, -y, -1, x * Y, y * Y, Y])
        A = np.asarray(A)
        _, _, Vt = np.linalg.svd(A)
        H = Vt[-1].reshape(3, 3)
        return H / H[2, 2]

    def _apply_homography(self, rgba: np.ndarray, H: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        h, w = target_shape
        ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        coords = np.stack([xs.ravel(), ys.ravel(), np.ones_like(xs).ravel()], axis=0)
        src = np.linalg.inv(H) @ coords
        src /= src[2:3]
        x_src = src[0].reshape(h, w)
        y_src = src[1].reshape(h, w)
        return self._bilinear_sample(rgba, x_src, y_src)

    def _bilinear_sample(self, image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        x0 = np.floor(x).astype(int)
        x1 = x0 + 1
        y0 = np.floor(y).astype(int)
        y1 = y0 + 1

        x0c = np.clip(x0, 0, w - 1)
        x1c = np.clip(x1, 0, w - 1)
        y0c = np.clip(y0, 0, h - 1)
        y1c = np.clip(y1, 0, h - 1)

        Ia = image[y0c, x0c]
        Ib = image[y1c, x0c]
        Ic = image[y0c, x1c]
        Id = image[y1c, x1c]

        wa = (x1 - x) * (y1 - y)
        wb = (x1 - x) * (y - y0)
        wc = (x - x0) * (y1 - y)
        wd = (x - x0) * (y - y0)

        sampled = Ia * wa[..., None] + Ib * wb[..., None] + Ic * wc[..., None] + Id * wd[..., None]
        mask = (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)
        sampled[~mask] = 0.0
        return sampled

    def _alpha_composite(self, base: np.ndarray, overlay: np.ndarray) -> np.ndarray:
        src_rgb, src_a = overlay[..., :3], overlay[..., 3:4]
        dst_rgb, dst_a = base[..., :3], base[..., 3:4]
        out_rgb = src_rgb * src_a + dst_rgb * (1.0 - src_a)
        out_a = src_a + dst_a * (1.0 - src_a)
        result = base.copy()
        result[..., :3] = out_rgb
        result[..., 3:4] = out_a
        return result

    def _downsample(self, image: np.ndarray, factor: int) -> np.ndarray:
        """Average-pool an oversampled image back to the viewport resolution."""

        if factor <= 1:
            return image
        h, w, c = image.shape
        new_h, new_w = h // factor, w // factor
        reshaped = image.reshape(new_h, factor, new_w, factor, c)
        return reshaped.mean(axis=(1, 3))

    def _smooth_gaze(self, gaze: GazeState) -> GazeState:
        """Low-pass filter the gaze command to suppress tiny head jitters."""

        if self.gaze_smoothing <= 0.0 or self._prev_gaze is None:
            self._prev_gaze = gaze
            return gaze

        alpha = self.gaze_smoothing
        prev = self._prev_gaze
        blended = GazeState(
            yaw_deg=prev.yaw_deg * alpha + gaze.yaw_deg * (1.0 - alpha),
            pitch_deg=prev.pitch_deg * alpha + gaze.pitch_deg * (1.0 - alpha),
            zoom=prev.zoom * alpha + gaze.zoom * (1.0 - alpha),
        )
        self._prev_gaze = blended
        return blended

    def _apply_temporal_smoothing(self, canvas: np.ndarray) -> np.ndarray:
        """Blend with the previous frame to reduce flicker in motion."""

        if self.temporal_smoothing <= 0.0 or self._prev_canvas is None:
            self._prev_canvas = canvas
            return canvas

        alpha = self.temporal_smoothing
        smoothed = self._prev_canvas * alpha + canvas * (1.0 - alpha)
        self._prev_canvas = smoothed
        return smoothed


def simulate_gaze_sequence(
    num_frames: int = 5,
    yaw_amplitude: float = 5.0,
    pitch_amplitude: float = 3.0,
) -> List[GazeState]:
    frames: List[GazeState] = []
    for idx in range(num_frames):
        t = idx / max(num_frames - 1, 1)
        yaw = math.sin(t * math.pi) * yaw_amplitude
        pitch = math.sin(t * math.pi * 0.5) * pitch_amplitude
        zoom = 1.0 + 0.05 * math.sin(t * math.pi)
        frames.append(GazeState(yaw_deg=yaw, pitch_deg=pitch, zoom=zoom))
    return frames
