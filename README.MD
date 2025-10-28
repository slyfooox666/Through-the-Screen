# CPSL Prototype — Content-Promoted Scene Layers

## 1. Overview

CPSL (**Content-Promoted Scene Layers**) is a new representation for dynamic 3D scenes designed to replace 3D Gaussian Splatting, NeRF, and point clouds in **high-efficiency, compact video applications**.

Instead of storing dense 3D primitives or implicit neural fields, CPSL encodes a scene as **a stack of semi-transparent 2D RGBA layers** aligned by high-fidelity **depth boundaries**.
Each layer is independently compressible with standard 2D video codecs (HEVC/AV1) and can be reprojected at runtime to produce **view-dependent parallax**.

This prototype implements the **complete CPSL pipeline**, optimized for **Video-on-Demand (VoD)** scenarios where all heavy preprocessing can run offline.
At playback, the system performs real-time view synthesis on commodity GPUs with minimal bandwidth and no 3D reconstruction.

---

## 2. Key Features

✅ **Depth-Driven Layer Fusion** – Geometry-only clustering yields crisp, object-aligned layers.
✅ **MV-Guided Temporal Propagation** – Reuses depth layer masks between frames via codec motion vectors for temporal stability.
✅ **Soft Alpha Bands** – Smooth boundary transitions to eliminate cracks or halos at layer edges.
✅ **Dynamic Pixel Strip (DPS)** – Runtime seam synthesis blends foreground/background to close residual cracks without touching static layers.
✅ **VoD-Optimized Storage** – Layers encoded as 2D RGBA streams, fully compatible with existing CDN/DASH infrastructure.
✅ **View-Adaptive Synthesizer** – Real-time parallax rendering from user pose without heavy neural inference.
✅ **Scalable Efficiency** – Adjustable number of layers (K), bitrate, and foveation level.
✅ **Depth Anything Integration** – Geometry-driven layer masks using high-capacity depth backbones.
✅ **Semantic-Free Pipeline** – The stack relies exclusively on depth cues; external segmentation stages were removed after they failed to preserve layer ordering.
✅ **Geometry-Aware Playback** – Plane-induced homographies per layer with linear-space compositing and optional Z-buffer.

---

## 3. System Architecture

┌────────────────────────────────────────────────────────────┐
│                Offline Preprocessing (VoD)                 │
│  1. Depth Estimation (Depth Anything)                      │
│  2. Depth Region Fusion   3. Motion-Vector Propagation     │
│  4. Encoding (HEVC/AV1) → VoD Storage (CDN)                │
└────────────────────────────────────────────────────────────┘
                 ↓                           ↑
┌────────────────────────────────────────────────────────────┐
│            Online Playback / View-Adaptive Synthesizer     │
│  - Load Layers (RGBA + Depth Metadata)                     │
│  - Compute Virtual Parallax per Layer (z_k)                │
│  - Homography Warp + Alpha Composite                       │
│  - Edge Processing (Soft Bands / Optional Filler)          │
│  - Output Novel View (6DoF Parallax)                       │
└────────────────────────────────────────────────────────────┘

---

## 4. Repository Layout

cpsl/
  configs/
    default.yaml             # configuration for preprocessing & playback
  data/
    input/                   # input videos
    output_vod/              # generated CPSL assets
  cpsl/
    models/                  # depth backends (MiDaS / Depth Anything)
    utils/                   # motion vectors, OBMC, warping, alpha, IO
    pipeline/                # keyframe fusion, propagation, encode VoD
    player/                  # view-adaptive synth + edge processing
  cpsl_preprocess.py          # offline preprocessor entrypoint
  cpsl_playback.py            # playback renderer
  cpsl_eval.py                # metrics & evaluation scripts
  tests/                      # unit tests for warping, alpha, OBMC

---

## 5. Installation

### Environment Setup
    git clone <repo-url>
    cd cpsl
    conda create -n cpsl python=3.10
    conda activate cpsl
    pip install -r requirements.txt

### Dependencies
- PyTorch ≥ 2.0 + CUDA (optional)
- Transformers + huggingface-hub (Depth Anything checkpoints)
- Depth Anything HF weights (or MiDaS fallback)
- Depth-Anything-V2 repository + local `.pth` checkpoints when operating offline
- ffmpeg / pyav (motion vectors)
- OpenCV, NumPy, SciPy, tqdm, PyYAML, LPIPS
- (Optional) PyOpenGL / ModernGL for GPU rendering

### Depth Model Setup (Offline)
- Clone `Depth-Anything-V2` into the project root (or provide a custom path via `preprocess.depth_repo`).
- Download the matching checkpoint (e.g., `depth_anything_v2_vitl.pth`) into the repo's `checkpoints/` folder.
- Update `configs/default.yaml` fields `depth_model`, `depth_checkpoint`, and `depth_repo` (paths are resolved relative to `configs/` so `../Depth-Anything-V2/...` targets the repo in the project root).

---

## 6. Configuration

`configs/default.yaml`
```
io:
  input_video: data/input/scene.mp4
  output_root: data/output_vod/scene_A
preprocess:
  gop: 12
  target_layers: 4
  depth_model: depth_anything_v2_vitl
  depth_checkpoint: ../Depth-Anything-V2/checkpoints/depth_anything_v2_vitl.pth
  depth_repo: ../Depth-Anything-V2
  depth_input_size: 518
  spatial_weight: 0.35
  depth_smooth_kernel: 7
  min_area_ratio: 0.004
  boundary_band_px: 5
  soft_alpha_sigma: 2.5
  soft_alpha_max: 0.95
  device: cuda
encode:
  codec: hevc
  crf: 18
  pix_fmt: yuva420p
playback:
  fps: 30
  viewer_intrinsics:
    fx: 1100
    fy: 1100
    cx: 960
    cy: 540
```
---

## 7. Usage Examples

### 1) Preprocess (Offline VoD Asset Generation)
```
python cpsl_preprocess.py --config configs/default.yaml
```
- or override settings directly:
```
python cpsl_preprocess.py \
  --input-video input/monkey.mp4 \
  --output-root data/output_vod/monkey \
  --depth-checkpoint ../Depth-Anything-V2/checkpoints/depth_anything_v2_vitl.pth \
  --depth-repo ../Depth-Anything-V2
```
Outputs:
- `output_vod/scene_A/frame_*/layer_*.png` (BGRA layers) + `_depth.npy`
- `metadata.json` with per-layer depth statistics and cluster metadata
- Encoded `.mp4` RGBA streams (HEVC/AV1)

### 2) Playback (View-Adaptive Rendering)
```
python cpsl_playback.py --config configs/default.yaml
```
- or override on the CLI:
```
python cpsl_playback.py \
  --input data/output_vod/monkey \
  --output data/output_vod/monkey/playback.mp4 \
  --trace traces/monkey_camera.csv \
  --max-view-offset 0.08 \
  --random-seed 42
```
`--input` should point to the preprocessing output directory (where `metadata.json` lives). 
`--trace` expects a CSV with `frame_index,tx,ty[,tz,yaw,pitch,roll]` (lines may be commented with `#`). Missing frames reuse the last provided pose. Without a trace the player synthesises a smooth random walk (seeded via `--random-seed`) instead of the old high-frequency jitter.
Controls:
- Mouse or keyboard to adjust yaw/pitch/roll.
- Optional foveation and pose control.
- Outputs parallax-corrected view frames in real time.
- Seam & crack sealing (`--crack-fix-*` flags):
  ```
  python cpsl_playback.py \
    --config configs/default.yaml \
    --input data/output_vod/monkey \
    --output data/output_vod/monkey/playback_crackfix.mp4 \
    --crack-fix-enabled \
    --crack-fix-band-px 3 \
    --crack-fix-dilate-px 2 \
    --crack-fix-depth-tol-rel 0.01 \
    --crack-fix-fill-strength 0.1 \
    --crack-fix-inpaint-radius 3 \
    --crack-fix-inpaint-alpha 0.2
  ```
  Leave additional controls (micro-offset sampling, depth jump gating, Z-buffer guard) at their defaults unless fine-tuning is required; disabling the flag reverts to the baseline compositor.

- Dynamic Pixel Strip is now on by default. Toggle or tune via:
  ```
  python cpsl_playback.py \
    --input data/output_vod/monkey \
    --output data/output_vod/monkey/playback_dps.mp4 \
    --dps.backend numpy \
    --dps.band_px 3 \
    --dps.feather_px 2
  ```
  Use `--dps.enable` to ensure DPS runs when overriding configs; set `--dps.backend cupy` to force GPU execution when available.

### 3) Evaluation
```
python cpsl_eval.py --metrics all
```
Reports:
- Boundary F-score (edge accuracy)
- Trimap IoU (alpha quality)
- Crack Rate (% low-opacity pixels at boundaries)
- LPIPS (band) (perceptual quality)
- Throughput (FPS during playback)

---

## 8. Expected Performance (1080p, K=6)

| Stage                 | Time/Frame        | Notes                         |
|-----------------------|-------------------|-------------------------------|
| Keyframe Preprocess   | 4–6 s (offline)   | heavy Mask2Former + depth     |
| MV Propagation        | 20–40 ms          | per P/B frame                 |
| Playback (Client)     | 3–5 ms            | GPU warp + blend              |
| Streaming Bitrate     | 5–12 Mbps         | VoD foveated playback         |

**Storage (1080p, 3 min, 8 viewports):** 2–5 GB total  
**Client GPU:** RTX 3060 / laptop iGPU runs real-time (60+ FPS)

---

## 9. Advantages over NeRF / 3DGS / Point Cloud

| Method         | Model Size     | Startup        | Playback Bandwidth | Rendering            |
|----------------|----------------|----------------|--------------------|----------------------|
| NeRF           | 10–50 GB       | Slow (train)   | Server-side        | Heavy MLP            |
| 3DGS           | 0.5–5 GB       | Slow (load)    | Low                | Fast raster but model-heavy |
| Point Cloud    | N/A            | Immediate      | >200 Mbps          | Sparse holes         |
| **CPSL (ours)**| 2–5 GB (VoD)   | Immediate      | 5–12 Mbps          | Pure 2D GPU ops      |

---

## 10. Results Visualization

Expected outputs:
- Sharp layer edges aligned with depth contours.
- Stable parallax when moving camera.
- No crack artifacts due to soft alpha bands.
- 1080p60 playback with <15 Mbps.

---

## 11. Future Extensions

- Add **learned boundary filler** (tiny UNet) for complex edges.
- Support **multi-view merging** for larger scenes.
- Integrate **live capture path** for MMSys (real-time CPSL generation).
- Extend crack-fix evaluation harness (crack-rate, boundary F-score, trimap IoU) and profile seam-pass runtime (goal ≤2.5 ms @1080p).

---

## 12. Citation

**CPSL: Content-Promoted Scene Layers for Compact Volumetric Video Representation**  
CVPR 2026 (submitted)  
Authors: [Your Name], [Collaborators]

---

## 13. Contact

Maintainer: [Your Name]  
Email: [your.email@domain.com]  
GitHub: [repo link]
