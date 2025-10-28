import time

import numpy as np
import pytest

from cpsl.player.dps import (
    DPSParams,
    LayerView,
    apply_dps,
    apply_dynamic_strip,
    build_dynamic_strip,
    reset_dps_state,
)
from cpsl.player.metrics import crack_rate
from cpsl.utils.img import from_premul, to_premul


@pytest.fixture(autouse=True)
def _reset_state():
    reset_dps_state()
    yield
    reset_dps_state()


def _make_layer(alpha_mask: np.ndarray, depth: float, color: np.ndarray, idx: int) -> LayerView:
    rgba = np.zeros(alpha_mask.shape + (4,), dtype=np.float32)
    rgba[..., :3] = color * alpha_mask[..., None]
    rgba[..., 3] = alpha_mask
    depth_map = np.full(alpha_mask.shape, depth, dtype=np.float32)
    mask = alpha_mask > 1e-5
    return LayerView(rgba=rgba, depth=depth_map, mask=mask, id=idx)


def test_seam_detection_generates_strip_mask():
    h, w = 32, 32
    alpha_front = np.zeros((h, w), dtype=np.float32)
    alpha_front[8:24, 8:23] = 1.0
    alpha_back = np.ones((h, w), dtype=np.float32)

    front = _make_layer(alpha_front, depth=0.5, color=np.array([1.0, 0.0, 0.0], dtype=np.float32), idx=0)
    back = _make_layer(alpha_back, depth=3.0, color=np.array([0.0, 1.0, 0.0], dtype=np.float32), idx=1)

    params = DPSParams(temporal_ema=0.0)
    strip_rgba, strip_mask = build_dynamic_strip([front, back], params)

    assert strip_rgba.shape == (h, w, 4)
    assert strip_mask.sum() > 0
    # Strip should be constrained near the foreground gap.
    assert strip_mask[:, 23:].sum() == 0


def test_apply_dynamic_strip_confined_to_mask():
    h, w = 16, 16
    base = np.zeros((h, w, 4), dtype=np.float32)
    strip = np.zeros_like(base)
    mask = np.zeros((h, w), dtype=bool)

    mask[4:8, 4:8] = True
    strip[mask, :3] = 0.5
    strip[mask, 3] = 0.6

    result = apply_dynamic_strip(base, strip, mask)
    assert np.allclose(result[~mask], base[~mask])
    assert np.allclose(result[mask, 3], 0.6)


def test_dynamic_strip_reduces_crack_rate():
    h, w = 32, 32
    alpha_front = np.zeros((h, w), dtype=np.float32)
    alpha_front[10:22, 10:22] = 1.0
    alpha_back = np.ones((h, w), dtype=np.float32) * 0.8
    alpha_back[12:20, 12:20] = 0.0  # mimic occluded background

    front = _make_layer(alpha_front, depth=0.5, color=np.array([1.0, 0.0, 0.0], dtype=np.float32), idx=0)
    back = _make_layer(alpha_back, depth=3.0, color=np.array([0.0, 0.0, 1.0], dtype=np.float32), idx=1)
    params = DPSParams(temporal_ema=0.0)

    base_rgba = np.dstack((back.rgba[..., :3] + front.rgba[..., :3], np.clip(alpha_back + alpha_front, 0.0, 1.0)))
    before = crack_rate(base_rgba[..., 3], band_px=3)

    strip_rgba, strip_mask = build_dynamic_strip([front, back], params)
    after_rgba = apply_dynamic_strip(base_rgba, strip_rgba, strip_mask)
    after = crack_rate(after_rgba[..., 3], band_px=3)

    assert after < before
    assert after <= before * 0.5


def test_apply_dps_background_reveal_prefers_deeper_layer():
    h, w = 32, 32
    alpha_front = np.zeros((h, w), dtype=np.float32)
    alpha_front[8:24, 4:16] = 1.0
    alpha_back = np.ones((h, w), dtype=np.float32)

    front_color = np.zeros((h, w, 3), dtype=np.float32)
    front_color[..., 0] = alpha_front
    back_color = np.zeros((h, w, 3), dtype=np.float32)
    back_color[..., 1] = alpha_back

    warped = {
        "color": np.stack([front_color, back_color], axis=0),
        "alpha": np.stack([alpha_front, alpha_back], axis=0),
        "depth": np.stack([np.full((h, w), 1.0, dtype=np.float32), np.full((h, w), 4.0, dtype=np.float32)], axis=0),
    }
    cfg = DPSParams(depth_tolerance=0.005, feather_weight=1.0, temporal_ema=0.0, band_px=3).to_config()

    result = apply_dps(warped, cfg=cfg)

    assert result["strip_mask"].sum() > 0
    assert result["bg_reveal_mask"].sum() > 0
    assert not result["fg_extension_mask"].any()
    green_energy = np.sum(result["strip_rgba"][..., 1])
    red_energy = np.sum(result["strip_rgba"][..., 0])
    assert green_energy > red_energy


def test_premult_roundtrip():
    rgb = np.array([[0.2, 0.4, 0.6]], dtype=np.float32)
    alpha = np.array([0.5], dtype=np.float32)
    rgba = to_premul(rgb, alpha)
    rgb_rec, alpha_rec = from_premul(rgba)
    assert np.allclose(alpha_rec, alpha, atol=1e-6)
    assert np.allclose(rgb_rec, rgb, atol=1e-6)


@pytest.mark.skipif(not hasattr(np, "float32"), reason="Requires NumPy float32 support.")
def test_dynamic_strip_performance_numpy():
    h, w = 180, 320
    rng = np.random.default_rng(0)
    alpha_front = (rng.random((h, w)) > 0.7).astype(np.float32)
    alpha_back = (rng.random((h, w)) > 0.3).astype(np.float32)

    front = _make_layer(alpha_front, depth=1.0, color=rng.random(3).astype(np.float32), idx=0)
    back = _make_layer(alpha_back, depth=4.0, color=rng.random(3).astype(np.float32), idx=1)
    params = DPSParams(temporal_ema=0.0)

    start = time.perf_counter()
    strip_rgba, strip_mask = build_dynamic_strip([front, back], params)
    elapsed = (time.perf_counter() - start) * 1000.0

    assert strip_rgba.shape == (h, w, 4)
    assert strip_mask.shape == (h, w)
    assert elapsed < 200.0
