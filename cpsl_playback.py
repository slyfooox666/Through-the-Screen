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
    parser.add_argument(
        "--crack-fix-enabled",
        action="store_true",
        help="Enable seam/crack filling during playback.",
    )
    parser.add_argument(
        "--crack-fix-band-px",
        type=int,
        default=None,
        help="Band radius (px) used to delineate seam area.",
    )
    parser.add_argument(
        "--crack-fix-dilate-px",
        type=int,
        default=None,
        help="Foreground dilation radius used for alpha growth.",
    )
    parser.add_argument(
        "--crack-fix-depth-tol-rel",
        type=float,
        default=None,
        help="Relative depth tolerance during dilation (fraction of depth range).",
    )
    parser.add_argument(
        "--crack-fix-micro-offset",
        type=float,
        default=None,
        help="Micro-offset in pixels for background pull sampling.",
    )
    parser.add_argument(
        "--crack-fix-fill-strength",
        type=float,
        default=None,
        help="Blend strength when borrowing colour from deeper layers.",
    )
    parser.add_argument(
        "--crack-fix-depth-jump-rel",
        type=float,
        default=None,
        help="Minimum relative depth jump required for background pull.",
    )
    parser.add_argument(
        "--crack-fix-inpaint-radius",
        type=int,
        default=None,
        help="Radius in pixels for Telea inpainting over residual cracks.",
    )
    parser.add_argument(
        "--crack-fix-inpaint-alpha",
        type=float,
        default=None,
        help="Alpha assigned to inpainted pixels inside the seam band.",
    )
    parser.add_argument(
        "--crack-fix-edge-alpha-threshold",
        type=float,
        default=None,
        help="Sobel magnitude threshold for alpha-based seam detection.",
    )
    parser.add_argument(
        "--crack-fix-edge-depth-threshold",
        type=float,
        default=None,
        help="Sobel magnitude threshold for depth-based seam detection.",
    )
    parser.add_argument(
        "--crack-fix-disable-zguard",
        action="store_true",
        help="Disable Z-buffer guard during crack fill compositing.",
    )
    parser.add_argument(
        "--debug-no-warp",
        action="store_true",
        help="Debug: disable homography warps (identity warp).",
    )
    parser.add_argument(
        "--dps.enable",
        dest="dps_enable",
        action="store_true",
        help="Enable Dynamic Pixel Strip seam synthesis.",
    )
    parser.add_argument(
        "--dps.band_px",
        dest="dps_band_px",
        type=int,
        default=None,
        help="Seam band width (pixels) for DPS.",
    )
    parser.add_argument(
        "--dps.feather_px",
        dest="dps_feather_px",
        type=int,
        default=None,
        help="Feather width (pixels) used by DPS.",
    )
    parser.add_argument(
        "--dps.max_pull_px",
        dest="dps_max_pull_px",
        type=int,
        default=None,
        help="Maximum dilation distance for foreground/background pulls.",
    )
    parser.add_argument(
        "--dps.depth_tolerance",
        dest="dps_depth_tolerance",
        type=float,
        default=None,
        help="Relative depth gap threshold distinguishing extension vs disocclusion.",
    )
    parser.add_argument(
        "--dps.alpha_threshold",
        dest="dps_alpha_threshold",
        type=float,
        default=None,
        help="Coverage threshold signalling uncovered regions in DPS.",
    )
    parser.add_argument(
        "--dps.feather_weight",
        dest="dps_feather_weight",
        type=float,
        default=None,
        help="Weight applied to feathered DPS contributions (0..1).",
    )
    parser.add_argument(
        "--dps.temporal_ema",
        dest="dps_temporal_ema",
        type=float,
        default=None,
        help="EMA factor for temporal DPS smoothing (0 disables).",
    )
    parser.add_argument(
        "--dps.extension_px",
        dest="dps_extension_px",
        type=float,
        default=None,
        help="Directional extension distance (pixels) used for foreground pulls.",
    )
    parser.add_argument(
        "--dps.inpaint_radius",
        dest="dps_inpaint_radius",
        type=int,
        default=None,
        help="Band-limited inpaint radius within the DPS seam.",
    )
    parser.add_argument(
        "--dps.inpaint",
        dest="dps_inpaint",
        type=str,
        default=None,
        help="Inpaint strategy for DPS gaps (telea|ns|none).",
    )
    parser.add_argument(
        "--dps.backend",
        dest="dps_backend",
        type=str,
        default=None,
        help="Backend for DPS computations (auto|numpy|cupy).",
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
    crack_fix_cfg = config.playback.crack_fix
    if args.crack_fix_enabled:
        crack_fix_cfg.enabled = True
    if args.crack_fix_band_px is not None:
        crack_fix_cfg.band_px = args.crack_fix_band_px
    if args.crack_fix_dilate_px is not None:
        crack_fix_cfg.dilate_px = args.crack_fix_dilate_px
    if args.crack_fix_depth_tol_rel is not None:
        crack_fix_cfg.depth_tol_rel = args.crack_fix_depth_tol_rel
    if args.crack_fix_micro_offset is not None:
        crack_fix_cfg.micro_offset_px = args.crack_fix_micro_offset
    if args.crack_fix_fill_strength is not None:
        crack_fix_cfg.fill_strength = args.crack_fix_fill_strength
    if args.crack_fix_depth_jump_rel is not None:
        crack_fix_cfg.depth_jump_rel = args.crack_fix_depth_jump_rel
    if args.crack_fix_inpaint_radius is not None:
        crack_fix_cfg.inpaint_radius = args.crack_fix_inpaint_radius
    if args.crack_fix_inpaint_alpha is not None:
        crack_fix_cfg.inpaint_alpha = args.crack_fix_inpaint_alpha
    if args.crack_fix_edge_alpha_threshold is not None:
        crack_fix_cfg.edge_alpha_threshold = args.crack_fix_edge_alpha_threshold
    if args.crack_fix_edge_depth_threshold is not None:
        crack_fix_cfg.edge_depth_threshold = args.crack_fix_edge_depth_threshold
    if args.crack_fix_disable_zguard:
        crack_fix_cfg.use_zbuffer_guard = False
    # Debug toggles attached to playback config for simplicity
    if args.debug_no_warp:
        setattr(config.playback, "debug_no_warp", True)
    dps_cfg = config.playback.dps
    if args.dps_enable:
        dps_cfg.enable = True
    if args.dps_band_px is not None:
        dps_cfg.band_px = args.dps_band_px
    if args.dps_feather_px is not None:
        dps_cfg.feather_px = args.dps_feather_px
    if args.dps_max_pull_px is not None:
        dps_cfg.max_pull_px = args.dps_max_pull_px
    if args.dps_depth_tolerance is not None:
        dps_cfg.depth_tolerance = args.dps_depth_tolerance
    if args.dps_alpha_threshold is not None:
        dps_cfg.alpha_threshold = args.dps_alpha_threshold
    if args.dps_feather_weight is not None:
        dps_cfg.feather_weight = args.dps_feather_weight
    if args.dps_temporal_ema is not None:
        dps_cfg.temporal_ema = args.dps_temporal_ema
    if args.dps_extension_px is not None:
        dps_cfg.extension_px = args.dps_extension_px
    if args.dps_inpaint_radius is not None:
        dps_cfg.inpaint_radius = args.dps_inpaint_radius
    if args.dps_inpaint is not None:
        dps_cfg.inpaint = args.dps_inpaint
    if args.dps_backend is not None:
        dps_cfg.backend = args.dps_backend
    output_path: Optional[Path] = args.output or config.playback.output_video
    if output_path is None:
        output_path = config.io.output_root / "playback.mp4"
    player = CPSLPlayer(config)
    player.render_to_video(output_path)


if __name__ == "__main__":
    main()
