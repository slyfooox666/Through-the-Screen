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
class PlaybackConfig:
    fps: float = 30.0
    viewer_intrinsics: ViewerIntrinsics = field(default_factory=ViewerIntrinsics)
    max_view_offset: float = 0.05  # meters
    output_video: Optional[Path] = None
    trace_path: Optional[Path] = None
    random_seed: Optional[int] = None


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
        playback_cfg = config.get("playback", {})

        def resolve_relative_to_base(value: Optional[str]) -> Optional[Path]:
            if value is None:
                return None
            path = Path(value)
            if path.is_absolute() or base_dir is None:
                return path
            return (base_dir / path)

        viewer_intrinsics_cfg = playback_cfg.get("viewer_intrinsics", {})
        playback_cfg = PlaybackConfig(
            fps=playback_cfg.get("fps", PlaybackConfig.__dataclass_fields__["fps"].default),
            viewer_intrinsics=ViewerIntrinsics(**viewer_intrinsics_cfg)
            if viewer_intrinsics_cfg
            else ViewerIntrinsics(),
            max_view_offset=playback_cfg.get(
                "max_view_offset",
                PlaybackConfig.__dataclass_fields__["max_view_offset"].default,
            ),
            output_video=resolve_relative_to_base(playback_cfg.get("output_video")),
            trace_path=resolve_relative_to_base(playback_cfg.get("trace_path")),
            random_seed=playback_cfg.get(
                "random_seed",
                PlaybackConfig.__dataclass_fields__["random_seed"].default,
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
