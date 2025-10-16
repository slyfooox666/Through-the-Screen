"""Command-line utilities for CPSL generation and rendering."""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import time
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from PIL import Image

try:  # Optional dependency for video handling
    import cv2
except Exception:  # pragma: no cover - handled at runtime
    cv2 = None

from .generator import Layer, LayeredFrameSet, MultiLayerGenerator

try:
    from .generator_sam import SAMMultiLayerGenerator
except Exception:  # pragma: no cover - optional dependency
    SAMMultiLayerGenerator = None
from .index import CPSLIndex
from .io_utils import ensure_dir, parse_time_spec, write_json
from .synthesizer import GazeState, LayeredFrameComposer


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _create_generator_from_args(args: argparse.Namespace) -> MultiLayerGenerator:
    seg_model = getattr(args, "segmentation_model", "deeplabv3_mobilenet_v3_large")
    seg_key = (seg_model or "").lower()

    if getattr(args, "no_semantics", False):
        seg_key = "none"

    if seg_key.startswith("sam"):
        if SAMMultiLayerGenerator is None:
            raise RuntimeError(
                "SAM generator not available. Install sam2 (for SAM 2.0) or segment-anything "
                "and ensure generator_sam.py dependencies are satisfied."
            )

        sam_checkpoint = getattr(args, "sam_checkpoint", None)
        sam_mode = getattr(args, "sam_mode", "local").lower()
        if sam_mode == "local" and not sam_checkpoint:
            raise ValueError("--sam-checkpoint is required when --sam-mode=local")

        sam_headers = getattr(args, "sam_headers", None)
        if sam_headers and isinstance(sam_headers, str):
            try:
                sam_headers = json.loads(sam_headers)
            except json.JSONDecodeError as exc:
                raise ValueError("--sam-headers must be valid JSON") from exc

        return SAMMultiLayerGenerator(
            sam_checkpoint=sam_checkpoint,
            sam_model=getattr(args, "sam_model", "sam2_hiera_t"),
            sam_mode=sam_mode,
            sam_endpoint=getattr(args, "sam_endpoint", None),
            sam_headers=sam_headers,
            max_sam_masks=getattr(args, "sam_max_masks", 5),
            num_layers=args.num_layers,
            depth_model_type=getattr(args, "depth_model", "DPT_Large"),
            device=getattr(args, "device", None),
            edge_softening_px=args.edge_softening,
            fill_occlusions=not getattr(args, "no_occlusion_fill", False),
            depth_smoothing_sigma=args.depth_smoothing,
            mask_morphology_radius=args.mask_morph_radius,
            semantic_confidence_threshold=getattr(args, "semantic_confidence", 0.25),
            semantic_depth_std_threshold=getattr(args, "semantic_depth_std", 0.08),
        )

    if seg_key in {"", "none"}:
        seg_model = None

    return MultiLayerGenerator(
        num_layers=args.num_layers,
        depth_model_type=getattr(args, "depth_model", "DPT_Large"),
        device=getattr(args, "device", None),
        edge_softening_px=args.edge_softening,
        fill_occlusions=not getattr(args, "no_occlusion_fill", False),
        depth_smoothing_sigma=args.depth_smoothing,
        mask_morphology_radius=args.mask_morph_radius,
        segmentation_model_type=seg_model,
        semantic_confidence_threshold=getattr(args, "semantic_confidence", 0.25),
        semantic_depth_std_threshold=getattr(args, "semantic_depth_std", 0.08),
    )


def _depth_to_uint16(depth: np.ndarray) -> np.ndarray:
    finite = np.isfinite(depth)
    if not finite.any():
        return np.zeros_like(depth, dtype=np.uint16)
    values = depth[finite]
    lo, hi = values.min(), values.max()
    if math.isclose(lo, hi):
        scaled = np.zeros_like(depth, dtype=np.float32)
    else:
        scaled = (depth - lo) / (hi - lo)
    scaled[~finite] = 0.0
    scaled = np.clip(scaled, 0.0, 1.0)
    return (scaled * 65535.0 + 0.5).astype(np.uint16)


def _write_layer(layer: Layer, layer_dir: Path) -> None:
    ensure_dir(layer_dir)
    rgba = np.clip(layer.rgba * 255.0, 0.0, 255.0).astype(np.uint8)
    Image.fromarray(rgba, mode="RGBA").save(layer_dir / "rgba.png")
    depth_u16 = _depth_to_uint16(layer.depth)
    Image.fromarray(depth_u16, mode="I;16").save(layer_dir / "depth.png")
    np.save(layer_dir / "depth.npy", layer.depth.astype("float32"), allow_pickle=False)
    np.save(layer_dir / "mask.npy", layer.mask.astype("bool"), allow_pickle=False)


def _write_metadata(
    frame_dir: Path,
    frame_index: int,
    pts_ms: float,
    layered: LayeredFrameSet,
    width: int,
    height: int,
    args: argparse.Namespace,
) -> None:
    fx = fy = max(width, height)
    cx = width / 2.0
    cy = height / 2.0
    semantic_model = getattr(args, "segmentation_model", None)
    semantic_model_key = (semantic_model or "").lower()
    semantics_enabled = not getattr(args, "no_semantics", False) and semantic_model_key not in {"", "none"}
    metadata = {
        "version": "0.1",
        "frame_index": int(frame_index),
        "pts_ms": float(pts_ms),
        "image_size": [width, height],
        "intrinsics": [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        "generator": {
            "num_layers": args.num_layers,
            "edge_softening": args.edge_softening,
            "depth_smoothing": args.depth_smoothing,
            "mask_morph_radius": args.mask_morph_radius,
            "depth_model": getattr(args, "depth_model", "DPT_Large"),
            "layer_scale_strength": getattr(args, "layer_scale_strength", 0.05),
            "layer_min_scale": getattr(args, "layer_min_scale", 1.0),
            "layer_parallax_min": getattr(args, "layer_parallax_min", 0.1),
            "semantic_model": semantic_model,
            "semantic_confidence": getattr(args, "semantic_confidence", None),
            "semantic_depth_std": getattr(args, "semantic_depth_std", None),
            "semantics_enabled": semantics_enabled,
            "sam_model": getattr(args, "sam_model", None),
            "sam_mode": getattr(args, "sam_mode", None),
            "sam_checkpoint": str(getattr(args, "sam_checkpoint", "")) if getattr(args, "sam_checkpoint", None) else None,
            "sam_endpoint": getattr(args, "sam_endpoint", None),
            "sam_max_masks": getattr(args, "sam_max_masks", None),
        },
        "order": "far_to_near",
        "alpha_premultiplied": True,
        "layers": [
            {
                "name": layer.name,
                **layer.statistics,
            }
            for layer in layered.layers
        ],
    }
    write_json(frame_dir / "metadata.json", metadata)


# ---------------------------------------------------------------------------
# Image command (legacy behaviour)
# ---------------------------------------------------------------------------


def _add_image_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("image", type=Path, help="Path to an RGB frame (PNG/JPG)")
    parser.add_argument(
        "--depth-map",
        type=Path,
        default=None,
        help="Optional depth map path (PNG or .npy). Skips MiDaS if provided.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory where layers and metadata will be written",
    )
    parser.add_argument("--num-layers", type=int, default=4, help="Number of depth bins")
    parser.add_argument(
        "--edge-softening",
        type=float,
        default=1.5,
        help="Gaussian blur radius (pixels) applied to alpha edges",
    )
    parser.add_argument("--no-occlusion-fill", action="store_true", help="Disable RGB inpainting")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Explicit torch device (e.g. 'cuda', 'cpu')",
    )
    parser.add_argument(
        "--depth-model",
        type=str,
        default="DPT_Large",
        help="MiDaS model id (e.g. DPT_Large, DPT_Hybrid, MiDaS_small)",
    )
    parser.add_argument(
        "--depth-smoothing",
        type=float,
        default=1.0,
        help="Gaussian sigma (pixels) applied to depth map; set to 0 to disable",
    )
    parser.add_argument(
        "--mask-morph-radius",
        type=int,
        default=1,
        help="Closing radius (pixels) to tidy per-layer masks; 0 disables",
    )
    parser.add_argument(
        "--segmentation-model",
        type=str,
        default="deeplabv3_mobilenet_v3_large",
        help="Semantic segmentation backbone (set to 'none' to disable)",
    )
    parser.add_argument(
        "--semantic-confidence",
        type=float,
        default=0.25,
        help="Confidence threshold for keeping semantic labels",
    )
    parser.add_argument(
        "--semantic-depth-std",
        type=float,
        default=0.08,
        help="Max depth std-dev within a semantic region before falling back to depth",
    )
    parser.add_argument(
        "--no-semantics",
        action="store_true",
        help="Disable semantic guidance during layer generation",
    )
    parser.add_argument(
        "--sam-checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path for SAM/SAM2 (required when --segmentation-model starts with 'sam')",
    )
    parser.add_argument(
        "--sam-model",
        type=str,
        default="sam2_hiera_t",
        help="SAM model identifier (e.g. sam2_hiera_t, sam_vit_h)",
    )
    parser.add_argument(
        "--sam-mode",
        choices=["local", "http"],
        default="local",
        help="Use local SAM inference or call a remote HTTP endpoint",
    )
    parser.add_argument(
        "--sam-endpoint",
        type=str,
        default=None,
        help="HTTP endpoint for SAM inference when --sam-mode=http",
    )
    parser.add_argument(
        "--sam-headers",
        type=str,
        default=None,
        help="JSON string of additional headers for SAM HTTP requests",
    )
    parser.add_argument(
        "--sam-max-masks",
        type=int,
        default=5,
        help="Number of top SAM masks to keep",
    )
    parser.add_argument("--save-composite", action="store_true", help="Write a composite preview PNG")
    return None


def _build_image_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate layered volumetric frame sets")
    _add_image_arguments(parser)
    return parser


def _command_image(parsed: argparse.Namespace) -> LayeredFrameSet:
    generator = _create_generator_from_args(parsed)

    depth = None
    if parsed.depth_map is not None:
        if parsed.depth_map.suffix == ".npy":
            depth = np.load(parsed.depth_map)
        else:
            depth = np.asarray(Image.open(parsed.depth_map))

    layered = generator.generate(parsed.image, depth_map=depth)

    output_dir = ensure_dir(parsed.output)
    for layer in layered.layers:
        layer_dir = output_dir / layer.name
        layer.save_debug_png(layer_dir)
        np.save(layer_dir / "depth.npy", layer.depth.astype("float32"), allow_pickle=False)
        np.save(layer_dir / "mask.npy", layer.mask.astype("bool"), allow_pickle=False)

    metadata = {
        "source": str(layered.source_image_path) if layered.source_image_path else None,
        "depth": layered.depth_metadata,
        "layers": [
            {
                "name": layer.name,
                **layer.statistics,
            }
            for layer in layered.layers
        ],
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as fout:
        json.dump(metadata, fout, indent=2)

    if parsed.save_composite:
        composite = layered.composite()
        Image.fromarray((composite * 255.0).astype("uint8"), mode="RGBA").save(
            output_dir / "composite.png"
        )

    return layered


# ---------------------------------------------------------------------------
# Video command
# ---------------------------------------------------------------------------


def _process_video(args: argparse.Namespace) -> None:
    if cv2 is None:
        raise RuntimeError("OpenCV is required for the 'video' command (pip install opencv-python)")

    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.input}")

    out_root = ensure_dir(args.out)
    # Clean stale artefacts from previous runs
    for existing in out_root.glob("frame_*"):
        if existing.is_dir():
            shutil.rmtree(existing)

    reassembled_root = out_root / "reassembled"
    if reassembled_root.exists():
        shutil.rmtree(reassembled_root)
    reassembled_root = ensure_dir(reassembled_root)

    index = CPSLIndex(out_root, mode="w")

    generator = _create_generator_from_args(args)
    composer = LayeredFrameComposer(
        layer_scale_strength=args.layer_scale_strength,
        layer_min_scale=args.layer_min_scale,
        layer_parallax_min=args.layer_parallax_min,
    )

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if args.start_ms > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, args.start_ms)
    if fps > 0:
        start_index = int(round(args.start_ms * fps / 1000.0))
    else:
        start_index = 0
    raw_frame_index = start_index
    monotonic_origin = None

    processed = 0
    written_frames = []
    t0 = time.perf_counter()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            pts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            if pts_ms <= 0:
                if fps > 0:
                    pts_ms = (raw_frame_index / fps) * 1000.0
                else:
                    if monotonic_origin is None:
                        monotonic_origin = time.monotonic()
                    pts_ms = (time.monotonic() - monotonic_origin) * 1000.0

            frame_index = raw_frame_index
            raw_frame_index += 1

            if args.end_ms >= 0 and pts_ms > args.end_ms:
                break

            if frame_index < start_index:
                continue

            if (frame_index - start_index) % args.frame_step != 0:
                continue

            tic = time.perf_counter()
            layered = generator.to_layers(frame)
            latency_ms = (time.perf_counter() - tic) * 1000.0

            frame_name = f"frame_{frame_index:06d}"
            frame_dir = out_root / frame_name
            if frame_dir.exists():
                shutil.rmtree(frame_dir)
            frame_dir = ensure_dir(frame_dir)
            layers_dir = ensure_dir(frame_dir / "layers")

            for layer in layered.layers:
                layer_path = layers_dir / layer.name
                _write_layer(layer, layer_path)

            _write_metadata(
                frame_dir,
                frame_index,
                pts_ms,
                layered,
                width=frame.shape[1],
                height=frame.shape[0],
                args=args,
            )

            if args.save_composite:
                composite = layered.composite()
                Image.fromarray((composite * 255.0).astype("uint8"), mode="RGBA").save(
                    frame_dir / "composite.png"
                )

            index.add(frame_index, pts_ms, frame_name, len(layered.layers))

            gaze = GazeState(
                yaw_deg=math.sin(processed * 0.1) * args.yaw_amplitude,
                pitch_deg=math.sin(processed * 0.07) * args.pitch_amplitude,
                zoom=1.0 + args.zoom_variation * math.sin(processed * 0.09),
            )
            composed = composer.compose_from_frame(frame_dir, gaze)
            render_path = reassembled_root / f"render_{processed:06d}.png"
            composed.save(render_path)
            written_frames.append(render_path)

            processed += 1

            print(
                f"frame {frame_name}: pts={pts_ms:9.3f} ms | layers={len(layered.layers)} | "
                f"latency={latency_ms:6.2f} ms"
            )
    finally:
        cap.release()
        index.close()

    elapsed = max(time.perf_counter() - t0, 1e-6)
    fps_out = processed / elapsed
    print(f"Processed {processed} frames in {elapsed:.2f}s ({fps_out:.2f} fps)")

    if args.write_video and written_frames:
        pattern = str((reassembled_root / "render_%06d.png").resolve())
        output_video = (out_root / "reassembled.mp4").resolve()
        command = [
            args.ffmpeg,
            "-y",
            "-framerate",
            str(args.framerate),
            "-i",
            pattern,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(output_video),
        ]
        print("Writing video via:", " ".join(command))
        subprocess.run(command, check=True)
        print("Video saved to", output_video)


# ---------------------------------------------------------------------------
# Render-at command
# ---------------------------------------------------------------------------


def _command_render_at(args: argparse.Namespace) -> None:
    index_path = Path(args.index)
    index = CPSLIndex(index_path.parent, index_path=index_path, mode="r")
    req_ms = parse_time_spec(args.at)
    record = index.nearest(req_ms, args.seek)

    frame_dir = index.root / record["relpath"]
    composer = LayeredFrameComposer()
    image = composer.compose_from_frame(
        frame_dir,
        GazeState(yaw_deg=args.yaw_deg, pitch_deg=args.pitch_deg, zoom=args.zoom),
    )
    out_path = Path(args.out)
    ensure_dir(out_path.parent)
    image.save(out_path)
    print(
        f"Rendered frame_index={record['frame_index']} pts_ms={record['pts_ms']:.3f} "
        f"-> {out_path}"
    )


# ---------------------------------------------------------------------------
# Stream command (placeholder)
# ---------------------------------------------------------------------------


def _command_stream(_: argparse.Namespace) -> None:
    print("Streaming ingest is not yet implemented. Coming soon.")


# ---------------------------------------------------------------------------
# Subcommand parser builders
# ---------------------------------------------------------------------------


def _add_video_subparser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("video", help="Convert a video into CPSL layers and renders")
    parser.add_argument("--input", required=True, type=Path, help="Input video (mp4)")
    parser.add_argument("--out", required=True, type=Path, help="Output directory root")
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--edge-softening", type=float, default=1.5)
    parser.add_argument("--depth-smoothing", type=float, default=1.0)
    parser.add_argument("--mask-morph-radius", type=int, default=1)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--depth-model", type=str, default="DPT_Large")
    parser.add_argument("--no-occlusion-fill", action="store_true")
    parser.add_argument("--segmentation-model", type=str, default="deeplabv3_mobilenet_v3_large")
    parser.add_argument("--semantic-confidence", type=float, default=0.25)
    parser.add_argument("--semantic-depth-std", type=float, default=0.08)
    parser.add_argument("--no-semantics", action="store_true")
    parser.add_argument("--sam-checkpoint", type=Path, default=None)
    parser.add_argument("--sam-model", type=str, default="sam2_hiera_t")
    parser.add_argument("--sam-mode", choices=["local", "http"], default="local")
    parser.add_argument("--sam-endpoint", type=str, default=None)
    parser.add_argument("--sam-headers", type=str, default=None)
    parser.add_argument("--sam-max-masks", type=int, default=5)
    parser.add_argument("--start-ms", type=int, default=0)
    parser.add_argument("--end-ms", type=int, default=-1)
    parser.add_argument("--save-composite", action="store_true")
    parser.add_argument("--write-video", action="store_true")
    parser.add_argument("--framerate", type=int, default=15)
    parser.add_argument("--ffmpeg", type=str, default="ffmpeg")
    parser.add_argument("--layer-scale-strength", type=float, default=0.05)
    parser.add_argument("--layer-min-scale", type=float, default=1.0)
    parser.add_argument("--layer-parallax-min", type=float, default=0.1)
    parser.add_argument("--yaw-amplitude", type=float, default=1.5)
    parser.add_argument("--pitch-amplitude", type=float, default=1.0)
    parser.add_argument("--zoom-variation", type=float, default=0.02)
    parser.set_defaults(func=_process_video)


def _add_render_subparser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("render-at", help="Render a novel view at a given timestamp")
    parser.add_argument("--index", required=True, type=Path, help="JSONL index path")
    parser.add_argument("--at", required=True, type=str, help="Timestamp (HH:MM:SS.mmm or NNNms)")
    parser.add_argument("--yaw-deg", type=float, default=0.0)
    parser.add_argument("--pitch-deg", type=float, default=0.0)
    parser.add_argument("--zoom", type=float, default=1.0)
    parser.add_argument("--out", required=True, type=Path, help="Output PNG path")
    parser.add_argument(
        "--seek",
        choices=["nearest", "floor", "ceil"],
        default="nearest",
        help="Seek strategy when locating timestamp",
    )
    parser.set_defaults(func=_command_render_at)


def _add_stream_subparser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("stream", help="Experimental streaming ingest (placeholder)")
    parser.add_argument("--source", type=str, default="0")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--edge-softening", type=float, default=1.5)
    parser.add_argument("--depth-smoothing", type=float, default=1.0)
    parser.add_argument("--mask-morph-radius", type=int, default=1)
    parser.add_argument("--segmentation-model", type=str, default="deeplabv3_mobilenet_v3_large")
    parser.add_argument("--semantic-confidence", type=float, default=0.25)
    parser.add_argument("--semantic-depth-std", type=float, default=0.08)
    parser.add_argument("--no-semantics", action="store_true")
    parser.add_argument("--window-sec", type=int, default=120)
    parser.add_argument("--index-type", type=str, default="jsonl")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.set_defaults(func=_command_stream)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CPSL tooling")
    subparsers = parser.add_subparsers(dest="command")
    _add_video_subparser(subparsers)
    _add_render_subparser(subparsers)
    _add_stream_subparser(subparsers)
    # Provide explicit image subcommand for completeness
    image_parser = subparsers.add_parser("image", help="(Legacy) generate layers from a single image")
    _add_image_arguments(image_parser)
    image_parser.set_defaults(func=lambda ns: _command_image(ns))
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        # Legacy behaviour: treat as single-image invocation.
        legacy_parser = _build_image_parser()
        legacy_args = legacy_parser.parse_args(argv)
        _command_image(legacy_args)
        return

    result = args.func(args)
    return result


def run_cli(args: Optional[argparse.Namespace] = None) -> LayeredFrameSet:
    """Legacy helper retained for backwards compatibility."""

    parser = _build_image_parser()
    parsed = parser.parse_args(args=args)
    return _command_image(parsed)


if __name__ == "__main__":  # pragma: no cover
    main()
