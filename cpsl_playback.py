#!/usr/bin/env python3
"""Entry point for CPSL playback prototype."""

from __future__ import annotations

import argparse
from pathlib import Path

from typing import Optional

from cpsl.config import load_yaml_config
from cpsl.player.playback import CPSLPlayer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render CPSL layers into a novel view video.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to YAML config file (defaults to configs/default.yaml).",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Override the CPSL output root containing metadata/layers.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional override for output video path.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override playback frame rate.",
    )
    parser.add_argument(
        "--max-view-offset",
        type=float,
        default=None,
        help="Override maximum view offset multiplier.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Seed for random camera trace generation when no trace file is provided.",
    )
    parser.add_argument(
        "--trace",
        type=Path,
        default=None,
        help="Optional CSV file with frame,x_offset,y_offset for camera trace.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    if args.input is not None:
        config.io.output_root = args.input
    if args.fps is not None:
        config.playback.fps = args.fps
    if args.max_view_offset is not None:
        config.playback.max_view_offset = args.max_view_offset
    if args.trace is not None:
        config.playback.trace_path = args.trace  # type: ignore[attr-defined]
    if args.random_seed is not None:
        config.playback.random_seed = args.random_seed  # type: ignore[attr-defined]
    output_path: Optional[Path] = args.output or config.playback.output_video
    if output_path is None:
        output_path = config.io.output_root / "playback.mp4"
    player = CPSLPlayer(config)
    player.render_to_video(output_path)


if __name__ == "__main__":
    main()
