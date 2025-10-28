# CPSL Prototype Context

## Overview
This project implements a **Content-Promoted Scene Layers (CPSL)** pipeline for representing dynamic 3D scenes in a compact, stream-friendly form.  
Instead of point clouds, 3D Gaussians, or NeRFs, CPSL decomposes each frame into a small stack of **depth-aligned RGBA layers** that can be encoded as ordinary 2D videos (HEVC/AV1).  
The system targets **Video-on-Demand (VoD)** playback, where heavy preprocessing is allowed offline and lightweight view-adaptive rendering runs in real time on the client.

---

## Pipeline Summary
### 1. Keyframe Preprocessing (offline)
- **Input:** RGB frames (or multi-view captures)
- **Depth Estimation:** Depth Anything V2 (local checkpoints) / MiDaS / DPT + confidence
- **Geometry Clustering:** combine depth + spatial cues to assign each pixel to K ordered layers
  (promote salient objects → foreground, merge background clusters)
- **Soft Alpha Bands:** distance-transform + adaptive width w = w₀ + a|∇z| + bσ_z
- **Metadata per layer:** { zₖ, nₖ, σ_zₖ, class, instance id }
- **Output:** premultiplied RGBA images + depth EXRs + metadata JSON.

### 2. Temporal Propagation (between keyframes)
- Extract codec **motion vectors (MVs)** from HEVC/AV1 bitstream.
- Convert to per-pixel motion field via OBMC (overlapped block motion compensation).
- Warp keyframe masks/alpha/depth to P/B frames with MVs.
- Depth-aware Z-buffer resolves occlusions; narrow-band edge refinement reduces blockiness.
- EMA smoothing maintains temporal stability.

### 3. Encoding for VoD
- Store per-viewport layer videos (RGBA or color + alpha) + metadata per segment.
- Layers are standard 2D streams → playable through existing CDN/DASH/HLS infra.

### 4. Client-Side View-Adaptive Synthesis (real time)
- Load current viewport’s layers and metadata.
- For each layer k:
  - Compute homography Hₖ = Kₜ (R – t nₖᵀ / zₖ) Kₛ⁻¹.
  - Warp color/alpha with per-layer homography (cv2.warpPerspective) and smooth edges.
- Composite in linear space front-to-back with optional Z-buffered occlusion.
- Produce parallax-corrected novel view per user pose.
- Optional gaze-conditioned gap filler runs on small boundary ROIs.

### 5. Metrics / Evaluation
- Boundary F-score, trimap IoU, crack-pixel rate, LPIPS (boundary band).
- Playback FPS and throughput for different K and bitrates.

---

## Efficiency Highlights
| Representation | Storage (1080p 3 min clip) | Playback Bitrate | Notes |
|----------------|----------------------------|------------------|-------|
| NeRF (dynamic) | 10–50 GB model + server render | Server-bound | Heavy compute |
| 3D GS/4DGS | 0.5–5 GB models | low | Large startup model |
| Point Cloud (V-PCC) | >200 Mbps | >200 Mbps | Bandwidth hungry |
| **CPSL (ours)** | 2–5 GB (all viewports) | 5–12 Mbps (1080p30 VoD) | 2D codecs, real-time client render |

---

## Key Advantages
- **Geometry-Only Layering:** high-fidelity depth masks without explicit semantics.
- **MV-Guided Temporal Propagation:** stable layers without re-segmenting.
- **Standard Codecs + VoD Workflow:** no custom streaming stack.
- **Real-time Client:** just 2D warps and alpha blends → < 5 ms per frame.
- **Scalable Representation:** parameter K layers × V viewports tunes quality vs. bitrate.

---

## Next Steps
- Improve depth-based clustering/regularisation for thin structures without relying on external semantics.
- Add learned boundary filler network (optional).
- Add evaluation script for bitrate/quality trade-off plots.
