# CPSL Prototype Milestone (Engineering Progress)

## Snapshot
- **Repository**: `CPSL/`
- **Goal**: Build an end-to-end Content-Promoted Scene Layers (CPSL) pipeline that converts monocular video into depth-ordered RGBA layers and replays them with lightweight parallax synthesis.
- **Prototype status**: Offline preprocessing and geometry-aware playback run end-to-end on sample clips; advanced VoD features (motion-vector propagation, codec integration, evaluation tooling) remain roadmap items.

## Implemented Components

### Configuration & Tooling
- YAML-driven configuration (`cpsl/config.py`) with dataclasses for IO, preprocessing, encoding placeholders, and playback parameters (intrinsics, traces, random walk control).
- CLI wrappers:
  - `cpsl_preprocess.py` orchestrates the offline pipeline and exposes overrides for all critical knobs (depth backend, layer budget, smoothing, device).
  - `cpsl_playback.py` loads generated assets, runs the synthesiser, and writes a novel-view video (with optional trace-driven camera paths).
- `configs/default.yaml` + `commandline.md` document out-of-the-box usage, while `PROJECT_DESCRIPTION.md` and `CONTEXT.MD` capture vision-level design intent.

### Depth & Layer Extraction
- `cpsl/models/depth.py` dispatches across MiDaS, Depth Anything (HF), and Depth Anything v2 checkpoints housed in `Depth-Anything-V2/`.
- `cpsl/pipeline/depth_regions.py` clusters `(depth, x, y)` features via k-means with spatial weighting and minimal area pruning.
- `cpsl/pipeline/layers.py` promotes clusters into CPSL layers with:
  - Soft alpha bands (Gaussian blur + clamp) to prevent edge cracks.
  - Automatic background / residual merging when cluster count exceeds the target layer budget.
- `cpsl/pipeline/preprocess.py` ties everything together:
  - Iterates video frames via `VideoReader`, predicts depth, clusters regions, and writes per-layer BGRA + depth maps.
  - Builds `metadata.json` capturing per-layer stats and relative paths.
- Sample outputs exist under `data/output_vod/{monkey, canoe, malaysia, porn}` validating the pipeline on real footage.

### Geometry-Aware Playback
- `cpsl/player/playback.py` loads metadata, generates camera poses (CSV trace or smoothed random walk), and feeds layers to the synthesiser.
- `cpsl/player/synth.py` provides the homography + compositing primitives:
  - Linear ↔ sRGB conversions, plane-induced homographies, edge-smoothed warps.
  - Z-buffer-aware front-to-back compositing to preserve occlusions.
- `cpsl/player/dps.py` has been rebuilt with a deterministic, geometry-aware Dynamic Pixel Strip (DPS):
  - Detects a thin seam band (edges + uncovered regions) and, per pixel, selects closest foreground and next deeper layer.
  - Uses local depth gap to choose foreground extension vs. true background reveal.
  - One-sided, premultiplied blending in linear space with optional band-limited inpainting.
  - Temporal stability via optional pose/flow reprojection + EMA blending.
  - Exposed as `apply_dps(warped: Dict[str,np.ndarray], D_star: Optional[np.ndarray], cfg: DpsConfig)` and integrated in the player; legacy `build_dynamic_strip` is preserved.
- `cpsl/player/edge_processing.py` provides an alternative band-limited crack filler (feature-gated).
- `VideoWriter` streams frames to disk; playback CLI supports FPS overrides, output selection, traces, and DPS knobs.

### Supporting Utilities
- `cpsl/utils/fs.py` and `cpsl/utils/video.py` wrap filesystem tasks and OpenCV IO.
- `cpsl/utils/img.py` and `cpsl/utils/morph.py` add premultiplication and morphology helpers backing DPS.
- Depth Anything repo (`Depth-Anything-V2/`) is vendored, enabling fully offline execution when checkpoints are supplied locally.
- `cpsl/player/metrics.py` provides crack rate, boundary F-score, and trimap IoU helpers for seam-fix ablations.

## Gaps & Open Work
- **Temporal propagation**: No motion-vector reuse or keyframe/P-frame handling yet—pipeline reprocesses every frame independently.
- **Encoding layer streams**: `EncodeConfig` is defined but unused; layers remain as PNG + `.npy` rather than HEVC/AV1 assets.
- **Evaluation tooling**: `cpsl_eval.py` remains a placeholder; automated reporting (LPIPS, crack metrics) needs wiring despite helper functions existing.
- **Normals / plane fitting**: Playback assumes fronto-parallel planes; normals are `None` and not persisted in metadata.
- **Unit tests / CI**: Add broader tests for depth dispatch, clustering, and synthesiser accuracy.
- **Semantic promotion**: `promote_classes` exists, but external segmentation was dropped; roadmap remains depth-driven grouping.

## Risks & Dependencies
- Dependence on large Depth Anything v2 checkpoints (stored outside repo) and CUDA availability.
- Lack of deterministic camera intrinsics in metadata; playback falls back to target intrinsics when source parameters are missing.
- Encoding step not wired into pipeline—VoD storage savings remain theoretical until integrated.
- Temporal coherence and disocclusion handling are minimal (single-frame soft alpha only).

## Recommended Next Moves
1. **Depth clustering quality**: Improve handling of thin structures/noise via smoothing/regularisation.
2. **Add temporal propagation**: Implement MV-guided warping between GOP keyframes and update metadata.
3. **Codec integration**: Wire `EncodeConfig` to ffmpeg/pyav to emit RGBA/alpha streams instead of PNGs.
4. **Evaluation harness**: Create `cpsl_eval.py` with automated metrics over `data/output_vod/*` for regressions.
5. **Metadata enrichment**: Persist normals/plane params to improve parallax accuracy.
6. **DPS profiling**: Benchmark rebuilt DPS; target <3 ms at 1080p, and tune defaults across scenes.
7. **Automation & tests**: Add smoke tests and CI to prevent regressions across depth backends and playback.

This milestone summary reflects the current engineering state of the CPSL prototype and should help prioritise the next development sprint.
