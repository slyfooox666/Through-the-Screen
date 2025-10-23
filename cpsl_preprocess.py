#!/usr/bin/env python3
"""Entry point for CPSL offline preprocessing."""

from __future__ import annotations

import argparse
from pathlib import Path

from cpsl.config import load_yaml_config
from cpsl.pipeline.preprocess import CPSLPreprocessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CPSL preprocessing pipeline.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to YAML config (defaults to configs/default.yaml).",
    )
    parser.add_argument("--input-video", type=Path, help="Override input video path.")
    parser.add_argument("--output-root", type=Path, help="Override output root directory.")
    parser.add_argument("--depth-model", type=str, help="Override depth model identifier.")
    parser.add_argument(
        "--depth-checkpoint",
        type=Path,
        help="Path to local Depth Anything V2 checkpoint (.pth).",
    )
    parser.add_argument(
        "--depth-repo",
        type=Path,
        help="Path to local Depth Anything V2 repository root.",
    )
    parser.add_argument(
        "--depth-input-size",
        type=int,
        help="Override depth model input size (pixels).",
    )
    parser.add_argument(
        "--target-layers",
        type=int,
        help="Override number of CPSL layers to extract per frame.",
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Torch device to use for inference (e.g. cuda, cuda:0, cpu).",
    )
    parser.add_argument(
        "--spatial-weight",
        type=float,
        help="Override spatial weighting used in depth clustering.",
    )
    parser.add_argument(
        "--depth-smooth-kernel",
        type=int,
        help="Override bilateral smoothing kernel size for depth clustering.",
    )
    parser.add_argument(
        "--min-area-ratio",
        type=float,
        help="Override minimum area ratio for retaining clustered regions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)

    if args.input_video is not None:
        config.io.input_video = args.input_video
    if args.output_root is not None:
        config.io.output_root = args.output_root
    preprocess_cfg = config.preprocess
    if args.depth_model is not None:
        preprocess_cfg.depth_model = args.depth_model
    if args.depth_checkpoint is not None:
        preprocess_cfg.depth_checkpoint = args.depth_checkpoint
    if args.depth_repo is not None:
        preprocess_cfg.depth_repo = args.depth_repo
    if args.depth_input_size is not None:
        preprocess_cfg.depth_input_size = args.depth_input_size
    if args.target_layers is not None:
        preprocess_cfg.target_layers = args.target_layers
    if args.device is not None:
        preprocess_cfg.device = args.device
    if args.spatial_weight is not None:
        preprocess_cfg.spatial_weight = args.spatial_weight
    if args.depth_smooth_kernel is not None:
        preprocess_cfg.depth_smooth_kernel = args.depth_smooth_kernel
    if args.min_area_ratio is not None:
        preprocess_cfg.min_area_ratio = args.min_area_ratio

    preprocessor = CPSLPreprocessor(config)
    preprocessor.run()


if __name__ == "__main__":
    main()
