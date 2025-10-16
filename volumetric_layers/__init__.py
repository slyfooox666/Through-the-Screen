"""Convenient entry points for the volumetric layer toolkit."""
from .generator import Layer, LayeredFrameSet, MultiLayerGenerator
from .index import CPSLIndex
from .io_utils import ensure_dir, parse_time_spec, write_json
from .synthesizer import GazeState, LayeredFrameComposer, simulate_gaze_sequence

__all__ = [
    "Layer",
    "LayeredFrameSet",
    "MultiLayerGenerator",
    "CPSLIndex",
    "ensure_dir",
    "parse_time_spec",
    "write_json",
    "GazeState",
    "LayeredFrameComposer",
    "simulate_gaze_sequence",
]
