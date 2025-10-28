"""Configuration dataclasses and utilities for the CPSL prototype."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class IOConfig:
    input_video: Path
    output_root: Path


@dataclass
class PreprocessConfig:
    gop: int = 12
    target_layers: int = 4
    sem_model: str = "deeplabv3_resnet50"
    depth_model: str = "DPT_Hybrid"
    promote_classes: List[str] = field(default_factory=list)
    boundary_band_px: int = 4
    soft_alpha_sigma: float = 2.5
    soft_alpha_max: float = 0.95
    device: str = "cuda"
    spatial_weight: float = 0.35
    depth_smooth_kernel: int = 5
    min_area_ratio: float = 0.005
    depth_checkpoint: Optional[Path] = None
    depth_repo: Optional[Path] = None
    depth_input_size: int = 518


@dataclass
class EncodeConfig:
    codec: str = "hevc"
    crf: int = 18
    pix_fmt: str = "yuva420p"


@dataclass
class ViewerIntrinsics:
    fx: float = 1100.0
    fy: float = 1100.0
    cx: float = 960.0
    cy: float = 540.0


@dataclass
class CrackFixConfig:
    enabled: bool = False
    band_px: int = 3
    dilate_px: int = 2
    depth_tol_rel: float = 0.01
    micro_offset_px: float = 0.5
    fill_strength: float = 0.3
    depth_jump_rel: float = 0.02
    inpaint_radius: int = 3
    inpaint_alpha: float = 0.2
    edge_alpha_threshold: float = 0.02
    edge_depth_threshold: float = 0.01
    use_zbuffer_guard: bool = True


@dataclass
class DPSConfig:
    enable: bool = False
    band_px: int = 3
    feather_px: int = 2
    max_pull_px: int = 2
    depth_tolerance: float = 0.02
    alpha_threshold: float = 0.98
    feather_weight: float = 0.65
    temporal_ema: float = 0.35
    extension_px: float = 1.5
    inpaint_radius: int = 2
    inpaint: str = "telea"
    backend: str = "auto"
    z_sigma: float = 0.01
    color_sigma: float = 0.1
    z_conf_thresh: float = 0.0


@dataclass
class PlaybackConfig:
    fps: float = 30.0
    viewer_intrinsics: ViewerIntrinsics = field(default_factory=ViewerIntrinsics)
    max_view_offset: float = 0.05  # meters
    output_video: Optional[Path] = None
    trace_path: Optional[Path] = None
    random_seed: Optional[int] = None
    crack_fix: CrackFixConfig = field(default_factory=CrackFixConfig)
    dps: DPSConfig = field(default_factory=DPSConfig)


@dataclass
class CPSLConfig:
    io: IOConfig
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    encode: EncodeConfig = field(default_factory=EncodeConfig)
    playback: PlaybackConfig = field(default_factory=PlaybackConfig)

    @staticmethod
    def from_dict(config: Dict[str, Any], base_dir: Optional[Path] = None) -> "CPSLConfig":
        """Build a CPSLConfig from nested dictionaries."""
        io_cfg = config.get("io", {})
        preprocess_cfg = dict(config.get("preprocess", {}))
        encode_cfg = config.get("encode", {})
        playback_cfg_dict = config.get("playback", {})

        def resolve_relative_to_base(value: Optional[str]) -> Optional[Path]:
            if value is None:
                return None
            path = Path(value)
            if path.is_absolute() or base_dir is None:
                return path
            return (base_dir / path)

        viewer_intrinsics_cfg = playback_cfg_dict.get("viewer_intrinsics", {})
        crack_fix_cfg = playback_cfg_dict.get("crack_fix", {})
        dps_cfg = playback_cfg_dict.get("dps", {})
        playback_cfg = PlaybackConfig(
            fps=playback_cfg_dict.get("fps", PlaybackConfig.__dataclass_fields__["fps"].default),
            viewer_intrinsics=ViewerIntrinsics(**viewer_intrinsics_cfg)
            if viewer_intrinsics_cfg
            else ViewerIntrinsics(),
            max_view_offset=playback_cfg_dict.get(
                "max_view_offset",
                PlaybackConfig.__dataclass_fields__["max_view_offset"].default,
            ),
            output_video=resolve_relative_to_base(playback_cfg_dict.get("output_video")),
            trace_path=resolve_relative_to_base(playback_cfg_dict.get("trace_path")),
            random_seed=playback_cfg_dict.get(
                "random_seed",
                PlaybackConfig.__dataclass_fields__["random_seed"].default,
            ),
            crack_fix=CrackFixConfig(
                enabled=crack_fix_cfg.get(
                    "enabled",
                    CrackFixConfig.__dataclass_fields__["enabled"].default,
                ),
                band_px=crack_fix_cfg.get(
                    "band_px",
                    CrackFixConfig.__dataclass_fields__["band_px"].default,
                ),
                dilate_px=crack_fix_cfg.get(
                    "dilate_px",
                    CrackFixConfig.__dataclass_fields__["dilate_px"].default,
                ),
                depth_tol_rel=crack_fix_cfg.get(
                    "depth_tol_rel",
                    CrackFixConfig.__dataclass_fields__["depth_tol_rel"].default,
                ),
                micro_offset_px=crack_fix_cfg.get(
                    "micro_offset_px",
                    CrackFixConfig.__dataclass_fields__["micro_offset_px"].default,
                ),
                fill_strength=crack_fix_cfg.get(
                    "fill_strength",
                    CrackFixConfig.__dataclass_fields__["fill_strength"].default,
                ),
                depth_jump_rel=crack_fix_cfg.get(
                    "depth_jump_rel",
                    CrackFixConfig.__dataclass_fields__["depth_jump_rel"].default,
                ),
                inpaint_radius=crack_fix_cfg.get(
                    "inpaint_radius",
                    CrackFixConfig.__dataclass_fields__["inpaint_radius"].default,
                ),
                inpaint_alpha=crack_fix_cfg.get(
                    "inpaint_alpha",
                    CrackFixConfig.__dataclass_fields__["inpaint_alpha"].default,
                ),
                edge_alpha_threshold=crack_fix_cfg.get(
                    "edge_alpha_threshold",
                    CrackFixConfig.__dataclass_fields__["edge_alpha_threshold"].default,
                ),
                edge_depth_threshold=crack_fix_cfg.get(
                    "edge_depth_threshold",
                    CrackFixConfig.__dataclass_fields__["edge_depth_threshold"].default,
                ),
                use_zbuffer_guard=crack_fix_cfg.get(
                    "use_zbuffer_guard",
                    CrackFixConfig.__dataclass_fields__["use_zbuffer_guard"].default,
                ),
            ),
            dps=DPSConfig(
                enable=dps_cfg.get("enable", DPSConfig.__dataclass_fields__["enable"].default),
                band_px=dps_cfg.get("band_px", DPSConfig.__dataclass_fields__["band_px"].default),
                feather_px=dps_cfg.get("feather_px", DPSConfig.__dataclass_fields__["feather_px"].default),
                max_pull_px=dps_cfg.get("max_pull_px", DPSConfig.__dataclass_fields__["max_pull_px"].default),
                depth_tolerance=dps_cfg.get(
                    "depth_tolerance",
                    DPSConfig.__dataclass_fields__["depth_tolerance"].default,
                ),
                alpha_threshold=dps_cfg.get(
                    "alpha_threshold",
                    DPSConfig.__dataclass_fields__["alpha_threshold"].default,
                ),
                feather_weight=dps_cfg.get(
                    "feather_weight",
                    DPSConfig.__dataclass_fields__["feather_weight"].default,
                ),
                temporal_ema=dps_cfg.get(
                    "temporal_ema",
                    DPSConfig.__dataclass_fields__["temporal_ema"].default,
                ),
                extension_px=dps_cfg.get(
                    "extension_px",
                    DPSConfig.__dataclass_fields__["extension_px"].default,
                ),
                inpaint_radius=dps_cfg.get(
                    "inpaint_radius",
                    DPSConfig.__dataclass_fields__["inpaint_radius"].default,
                ),
                z_sigma=dps_cfg.get("z_sigma", DPSConfig.__dataclass_fields__["z_sigma"].default),
                color_sigma=dps_cfg.get("color_sigma", DPSConfig.__dataclass_fields__["color_sigma"].default),
                z_conf_thresh=dps_cfg.get("z_conf_thresh", DPSConfig.__dataclass_fields__["z_conf_thresh"].default),
                inpaint=dps_cfg.get("inpaint", DPSConfig.__dataclass_fields__["inpaint"].default),
                backend=dps_cfg.get("backend", DPSConfig.__dataclass_fields__["backend"].default),
            ),
        )

        if "depth_checkpoint" in preprocess_cfg:
            preprocess_cfg["depth_checkpoint"] = resolve_relative_to_base(preprocess_cfg.get("depth_checkpoint"))
        if "depth_repo" in preprocess_cfg:
            preprocess_cfg["depth_repo"] = resolve_relative_to_base(preprocess_cfg.get("depth_repo"))

        return CPSLConfig(
            io=IOConfig(
                input_video=resolve_relative_to_base(io_cfg["input_video"]),
                output_root=resolve_relative_to_base(io_cfg["output_root"]),
            ),
            preprocess=PreprocessConfig(**preprocess_cfg),
            encode=EncodeConfig(**encode_cfg),
            playback=playback_cfg,
        )


def load_yaml_config(path: Path) -> CPSLConfig:
    """Load a YAML configuration file into a CPSLConfig."""
    import yaml

    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return CPSLConfig.from_dict(raw, base_dir=path.parent)
