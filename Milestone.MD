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
- `cpsl/player/synth.py` provides all homography and compositing primitives:
  - Linear ↔ sRGB conversions, plane-induced homographies, edge-smoothed warps.
  - Z-buffer-aware front-to-back compositing to preserve occlusions.
- `cpsl/player/dps.py` implements the Dynamic Pixel Strip (DPS) runtime seam filler, blending foreground/background layers inside a narrow band before final composite.
- `cpsl/player/edge_processing.py` (new) introduces a band-limited crack filler that performs depth-aware dilation, background pulls, and Telea inpainting, exposed via config/CLI flags and wired into playback behind a feature gate.
- `VideoWriter` utility streams resulting frames to disk; playback CLI supports FPS overrides and output path selection.

### Supporting Utilities
- `cpsl/utils/fs.py` and `cpsl/utils/video.py` wrap filesystem tasks and OpenCV IO.
- `cpsl/utils/img.py` and `cpsl/utils/morph.py` add premultiplication and morphology helpers backing DPS.
- Depth Anything repo (`Depth-Anything-V2/`) is vendored, enabling fully offline execution when checkpoints are supplied locally.
- `cpsl/player/metrics.py` now provides crack rate, boundary F-score, and trimap IoU helpers for seam-fix ablations.

## Gaps & Open Work
- **Temporal propagation**: No motion-vector reuse or keyframe/P-frame handling yet—pipeline reprocesses every frame independently.
- **Encoding layer streams**: `EncodeConfig` is defined but unused; layers remain as PNG + `.npy` rather than HEVC/AV1 assets.
- **Evaluation tooling**: `cpsl_eval.py` is referenced in docs but absent; automated reporting (LPIPS, crack metrics) still needs wiring despite helper functions existing.
- **Normals / plane fitting**: Playback currently assumes fronto-parallel planes; normals are `None` and not persisted in metadata.
- **Unit tests / CI**: No automated validation covering depth dispatch, clustering, or synthesiser accuracy.
- **Semantic promotion**: `promote_classes` hook exists, but external segmentation was dropped after it failed to provide reliable depth ordering; current roadmap focuses on depth-driven grouping.

## Risks & Dependencies
- Dependence on large Depth Anything v2 checkpoints (stored outside repo) and CUDA availability.
- Lack of deterministic camera intrinsics in metadata; playback falls back to target intrinsics when source parameters are missing.
- Encoding step not wired into pipeline—VoD storage savings remain theoretical until integrated.
- Temporal coherence and disocclusion handling are minimal (single-frame soft alpha only).

## Recommended Next Moves
1. **Depth clustering quality**: Investigate depth-only failure cases (thin structures, noisy maps) via improved smoothing/regularisation rather than external semantics.
2. **Add temporal propagation**: Implement motion-vector warping between GOP keyframes and update metadata schema accordingly.
3. **Codec integration**: Hook `EncodeConfig` into an ffmpeg/pyav encoder to emit RGBA/alpha streams instead of standalone PNGs.
4. **Evaluation harness**: Create `cpsl_eval.py` with automated metrics over `data/output_vod/*`, enabling regression tracking.
5. **Metadata enrichment**: Persist estimated normals/plane parameters to improve playback parallax accuracy.
6. **Crack-fix validation**: Wire the new metrics into ablation scripts (A0–A3) and profile runtime to stay within the ≤2.5 ms budget.
7. **Automation & tests**: Introduce smoke tests (short clip) and CI scripts to prevent regressions across depth backends and playback.

This milestone summary reflects the current engineering state of the CPSL prototype and should help prioritise the next development sprint.
