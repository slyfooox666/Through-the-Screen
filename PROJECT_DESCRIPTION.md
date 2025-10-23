# CPSL Prototype Code Overview

This document summarises the current CPSL (Content-Promoted Scene Layers) prototype, covering the pipeline stages, key modules, configuration surfaces, and command-line tooling. It is intended as a reference when writing documentation or publications about the system.

---

## 1. High-Level Pipeline

1. **Offline Preprocessing (`cpsl/pipeline/preprocess.py`)**
   - Loads a source video.
   - Runs Depth Anything v2 (or MiDaS fallback) to estimate dense depth maps.
   - Clusters depth into a fixed number of layers via `cluster_depth_regions`.
   - Generates premultiplied RGBA layer assets, warped depth maps, and metadata.
   - Persists everything under `io.output_root` with a master `metadata.json`.

2. **Geometry-Aware Playback (`cpsl/player/playback.py` + `cpsl/player/synth.py`)**
   - Reads layer assets and metadata.
   - Builds source/target intrinsics, camera pose traces (random walk or CSV).
   - For each layer: computes a plane-induced homography, warps colour/alpha/depth, smooths edges.
   - Performs linear-space front-to-back compositing with optional Z-buffering.
   - Writes the resulting video to disk.

---

## 2. Core Modules

### 2.1 Configuration (`cpsl/config.py`)
- Dataclasses for IO, preprocessing, encoding, and playback.
- YAML loader (`load_yaml_config`) with relative-path resolution.
- Playback config recognises target intrinsics, optional trace path, random seed.

### 2.2 Depth Models (`cpsl/models/depth.py`)
- Dispatcher supporting MiDaS (`_MiDaSBackend`), Depth Anything HF checkpoints (`_DepthAnythingBackend`), and Depth Anything v2 local checkpoints (`_DepthAnythingV2Backend`).
- `DepthEstimator.predict` returns normalised depth maps shaped like the input frame.

### 2.3 Layer Generation (`cpsl/pipeline/layers.py`, `cpsl/pipeline/depth_regions.py`)
- `cluster_depth_regions` clusters `(depth, x, y)` features with k-means, sorts by depth.
- `generate_layers` applies soft alpha bands, saves per-layer stats.
- Layers default to geometry-only ordering; a background/residual layer is appended if needed.

### 2.4 Preprocessing Orchestration (`cpsl/pipeline/preprocess.py`)
- Wraps depth inference + clustering.
- Writes layer textures (`layer_##.png`) and depth maps (`layer_##_depth.npy`).
- Collects per-frame metadata (`FrameMetadata`, `FrameLayerMetadata`) into `metadata.json`.

### 2.5 Geometry Synthesiser (`cpsl/player/synth.py`)
- Intrinsic matrix helper (`build_intrinsic_matrix`).
- Plane-induced homography (`plane_homography`).
- sRGB↔linear conversions for accurate alpha blending.
- `warp_layer` applies `cv2.warpPerspective` to colour/alpha/depth with edge smoothing.
- `composite_front_to_back` implements premultiplied blending and Z-buffer occlusion.
- `render_frame` runs the full pipeline and returns a BGR frame.

### 2.6 Playback Controller (`cpsl/player/playback.py`)
- Loads metadata, builds camera pose sequences.
- Supports CSV traces with translation (`tx, ty, tz`) + Euler rotations (yaw, pitch, roll).
- Generates smoothed random walks when no trace is supplied.
- Computes per-layer plane depths (using inverse depth for correct ordering).
- Calls `synth.render_frame` and streams frames via `VideoWriter`.

---

## 3. Command-Line Interfaces

### 3.1 Preprocessing (`cpsl_preprocess.py`)
- Arguments:
  - `--config` (default `configs/default.yaml`).
  - Overrides: `--input-video`, `--output-root`, `--depth-model`, `--depth-checkpoint`, `--depth-repo`, `--depth-input-size`, `--target-layers`, `--device`, `--spatial-weight`, `--depth-smooth-kernel`, `--min-area-ratio`.
- Typical usage:
  ```bash
  python cpsl_preprocess.py \
    --input-video input/monkey.mp4 \
    --output-root data/output_vod/monkey \
    --depth-repo ../Depth-Anything-V2 \
    --depth-checkpoint ../Depth-Anything-V2/checkpoints/depth_anything_v2_vitl.pth
  ```
- Output artefacts:
  - `frame_XXXX/layer_##.png` (BGRA).
  - `frame_XXXX/layer_##_depth.npy`.
  - `metadata.json` (per-frame layer descriptors, video metadata, intrinsics placeholder).

### 3.2 Playback (`cpsl_playback.py`)
- Arguments:
  - `--config` (default `configs/default.yaml`).
  - `--input` (override `io.output_root`).
  - `--output` (video path).
  - `--fps`, `--max-view-offset`.
  - `--trace` (CSV with `frame,tx,ty[,tz,yaw,pitch,roll]`).
  - `--random-seed`.
- Example with geometry-aware rendering:
  ```bash
  python cpsl_playback.py \
    --input data/output_vod/monkey \
    --output data/output_vod/monkey/playback.mp4 \
    --max-view-offset 0.03 \
    --random-seed 42
  ```

---

## 4. Data Layout

```
CPSL/
├── configs/
│   └── default.yaml                # example configuration
├── cpsl/
│   ├── __init__.py
│   ├── config.py
│   ├── models/                     # depth backends
│   ├── pipeline/
│   │   ├── preprocess.py           # offline pipeline orchestrator
│   │   ├── layers.py               # layer generation utilities
│   │   └── depth_regions.py        # depth clustering helpers
│   ├── player/
│   │   ├── __init__.py
│   │   ├── playback.py             # geometry-aware playback controller
│   │   └── synth.py                # homography + compositing logic
│   └── utils/
│       ├── fs.py
│       └── video.py
├── cpsl_preprocess.py              # CLI entrypoint (offline)
├── cpsl_playback.py                # CLI entrypoint (playback)
├── PROJECT_DESCRIPTION.md          # (this document)
├── README.MD
└── CONTEXT.MD
```

---

## 5. Key Design Notes

- **Geometry-Only Layering:** No semantic segmentation is required; layer boundaries are derived from depth clustering.
- **Depth Anything v2 Integration:** Supports offline checkpoints cloned from the DepthAnything-V2 repository, controlled via config.
- **Linear Colour Pipeline:** All blending happens in linear RGB to avoid brightening/darkening artefacts at layer edges.
- **Homography-Based Warping:** Each layer is treated as a fronto-parallel plane with homography `H_k`. Nearer layers naturally exhibit stronger parallax.
- **Z-Buffer Occlusion:** Optional Z-buffer ensures front layers occlude properly even when warped regions overlap.
- **Trace-Driven View Paths:** Camera motion can be deterministic (CSV trace) or procedurally generated (smoothed random walk constrained by `max_view_offset`).
- **Extensible Metadata:** `metadata.json` currently records depth statistics per layer; the schema is ready for normals / confidence maps if future stages require them.

---

## 6. Future Extension Points

1. **Normals & Variance:** Store per-layer normals/uncertainty and feed them into `plane_homography` for slanted surfaces.
2. **Temporal Tracking:** Use codec motion vectors to propagate layers frame-to-frame for streaming scenarios.
3. **Inpainting:** Add learned or procedural hole filling where warps expose disoccluded regions.
4. **Evaluation Scripts:** Implement metrics (LPIPS, trimap IoU, boundary F-score) in `cpsl_eval.py`.
5. **Interactive Viewer:** Hook `render_frame` into a real-time UI (OpenGL/ModernGL) driven by head tracking.

---

## 7. References for Paper Writing

- **Algorithm:** Section 2 and 3 highlight the algorithmic flow: depth clustering, layer generation, homography-based synthesis.
- **Implementation Details:**
  - Depth estimator dispatch (mixture of HF and local models).
  - Layer masks with soft-alpha boundary smoothing.
  - Linear-space compositing with optional Z-buffer.
  - CLI workflow enabling reproducible experiments.
- **Performance Considerations:** Preprocessing is GPU-heavy (Depth Anything). Playback is lightweight—per-frame cost is dominated by homography warps and alpha blending.

Use this overview as a blueprint when drafting the project paper or preparing slides; it captures the code’s current architecture and the main contributions of the CPSL prototype.

