"""Convert a short video into CPSL layers and gaze-adaptive renderings."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image

from volumetric_layers import GazeState, LayeredFrameComposer, MultiLayerGenerator


def save_layered_frame(frame_dir: Path, layered) -> None:
    frame_dir.mkdir(parents=True, exist_ok=True)
    for layer in layered.layers:
        layer_dir = frame_dir / layer.name
        layer_dir.mkdir(parents=True, exist_ok=True)
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
    with (frame_dir / "metadata.json").open("w", encoding="utf-8") as fout:
        json.dump(metadata, fout, indent=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert a video into CPSL layers + renderings")
    parser.add_argument("video", type=Path, help="Input video path")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/video_run"),
        help="Output directory root",
    )
    parser.add_argument("--num-layers", type=int, default=4, help="Layers per frame")
    parser.add_argument(
        "--edge-softening",
        type=float,
        default=1.5,
        help="Alpha edge softening radius (px)",
    )
    parser.add_argument(
        "--depth-model",
        type=str,
        default="DPT_Large",
        help="MiDaS model id",
    )
    parser.add_argument(
        "--depth-smoothing",
        type=float,
        default=1.0,
        help="Depth smoothing sigma (px)",
    )
    parser.add_argument(
        "--mask-morph-radius",
        type=int,
        default=1,
        help="Mask morphology radius (px)",
    )
    parser.add_argument("--device", type=str, default=None, help="Torch device override")
    parser.add_argument("--no-occlusion-fill", action="store_true", help="Disable RGB inpainting")
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
        help="Confidence threshold for semantic guidance",
    )
    parser.add_argument(
        "--semantic-depth-std",
        type=float,
        default=0.08,
        help="Maximum depth std-dev for semantic grouping",
    )
    parser.add_argument("--no-semantics", action="store_true", help="Disable semantic guidance")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Process at most N frames",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Process every Nth frame (1 = all frames)",
    )
    parser.add_argument(
        "--yaw-amplitude",
        type=float,
        default=5.0,
        help="Yaw sweep amplitude in degrees",
    )
    parser.add_argument(
        "--pitch-amplitude",
        type=float,
        default=3.0,
        help="Pitch sweep amplitude in degrees",
    )
    parser.add_argument(
        "--zoom-variation",
        type=float,
        default=0.05,
        help="Zoom variation fraction (0.05 => ±5%)",
    )
    parser.add_argument(
        "--write-video",
        action="store_true",
        help="Emit a reassembled MP4 (requires ffmpeg on PATH)",
    )
    parser.add_argument(
        "--framerate",
        type=int,
        default=15,
        help="Framerate for the optional output video",
    )
    parser.add_argument(
        "--ffmpeg",
        type=str,
        default="ffmpeg",
        help="ffmpeg executable when --write-video is set",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.video.exists():
        raise FileNotFoundError(args.video)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.video}")

    output_root = args.output
    layers_root = output_root / "layers"
    reassembled_root = output_root / "reassembled"
    layers_root.mkdir(parents=True, exist_ok=True)
    reassembled_root.mkdir(parents=True, exist_ok=True)

    seg_model = args.segmentation_model
    if seg_model and seg_model.lower() == "none":
        seg_model = None
    if args.no_semantics:
        seg_model = None

    generator = MultiLayerGenerator(
        num_layers=args.num_layers,
        depth_model_type=args.depth_model,
        device=args.device,
        edge_softening_px=args.edge_softening,
        fill_occlusions=not args.no_occlusion_fill,
        depth_smoothing_sigma=args.depth_smoothing,
        mask_morphology_radius=args.mask_morph_radius,
        segmentation_model_type=seg_model,
        semantic_confidence_threshold=args.semantic_confidence,
        semantic_depth_std_threshold=args.semantic_depth_std,
    )
    composer = LayeredFrameComposer()

    frame_idx = 0
    processed_idx = 0
    written_frames: List[Path] = []

    print(f"Processing video: {args.video} → {output_root}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % args.frame_step != 0:
            frame_idx += 1
            continue
        if args.max_frames is not None and processed_idx >= args.max_frames:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        layered = generator.generate(rgb)

        frame_name = f"frame_{processed_idx:04d}"
        frame_dir = layers_root / frame_name
        save_layered_frame(frame_dir, layered)

        if processed_idx == 0:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            total_frames = total_frames if total_frames > 0 else processed_idx + 1
        angle = (processed_idx / max(args.max_frames or total_frames, 1)) * math.pi
        yaw = math.sin(angle) * args.yaw_amplitude
        pitch = math.sin(angle * 0.5) * args.pitch_amplitude
        zoom = 1.0 + args.zoom_variation * math.sin(angle)
        gaze = GazeState(yaw_deg=yaw, pitch_deg=pitch, zoom=zoom)

        rgba_layers = [layer.rgba for layer in layered.layers]
        depth_layers = [layer.depth for layer in layered.layers]
        composed = composer.compose_layers(rgba_layers, depth_layers, gaze)
        out_path = reassembled_root / f"{frame_name}.png"
        composed.save(out_path)
        written_frames.append(out_path)

        print(f"Processed {frame_name}: yaw={yaw:.2f}°, pitch={pitch:.2f}°, zoom={zoom:.3f}")

        frame_idx += 1
        processed_idx += 1

    cap.release()

    if args.write_video and written_frames:
        pattern = str((reassembled_root / "frame_%04d.png").resolve())
        output_video = (output_root / "reassembled.mp4").resolve()
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

    print(f"Completed {processed_idx} frames.")


if __name__ == "__main__":
    main()
