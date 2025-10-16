# Project Context for Submission to CVPR 2026 -- CPSL: Representing Volumetric Video via Content-Promoted Scene Layers

## Research Goal
- This project proposes **CPSL (Content-Promoted Scene Layers)**, a novel 3D scene representation method.  
- Core idea: Based on content semantics and saliency, construct multiple Scene Layers where important regions are represented and transmitted with higher fidelity, while background or less relevant regions are more strongly compressed. The output of the representation is sets of layers at specific ovewports, and during the playback process, the layers are re-assembled according to the user's actual viewport, each layer could be adjusted in size and angles (ptich, yaw); in this way, we render a 3D scene by using several pre-constructed layers to replace traditional heavy 3D representations like 3D GS or NeRF or point clouds.
- Objective: Guarantee **high visual quality (especially in ROI regions)** while significantly improving **compression efficiency**, outperforming traditional 3D representations such as NeRF and 4DGS.

## Key Contributions
1. **CPSL Representation**  
   - We introduce **Content-Promoted Scene Layers (CPSL)** as a novel 3D scene representation tailored for high-quality and efficient video transmission.  

2. **Content-Promoted Layer Extraction**  
   - We design a saliency- and semantics-aware extraction pipeline that allocates bitrate budget adaptively, ensuring critical content is promoted to higher-fidelity layers.  

3. **Gaze-Conditioned Reassembly (Offline Evaluation)**  
   - We demonstrate that CPSL can be reassembled under gaze-conditioned settings, highlighting the adaptability of our representation to perceptual prioritization.  

4. **Comprehensive Comparisons**  
   - We provide extensive experimental results against strong baselines including NeRF, 3D Gaussian Splatting (3DGS), and point cloud renderers, showing that CPSL achieves superior trade-offs in quality and compression efficiency.  

## Modules Implementation
- Multi-layer Generator 
- Layer Synthesizer
- Lightweight semantic segmentation (DeepLabV3 MobileNet) guides layer
  assignments to preserve content edges during CPSL extraction.

## System Constraints
- ROI error tolerance: ΔPSNR < 0.5 dB (to be updated)
- Global error tolerance: ΔPSNR < 1.5 dB  (to be updated) 
- Compression ratio: ≥ 2× compared with NeRF baseline  (to be updated) 
- Decoding latency: relaxed (target < 100 ms/frame, higher allowed for offline settings)  
- Maximum number of layers: ≤ 5, with differences determined by content saliency rather than uniform segmentation

## Repository Structure
- `README.md` – quickstart setup and minimal usage instructions.
- `CONTEXT.md` – canonical project description (this document).
- `input/` – optional folder to stash source RGB or RGB-D frames.
- `output/` – generated layered assets and synthesized views (git-ignored).
- `scripts/`
  - `render_gaze_demo.py` – renders a small gaze sweep using precomputed layers.
- `src/`
  - `volumetric_layers/`
    - `__init__.py` – package exports (`MultiLayerGenerator`, `LayeredFrameComposer`, etc.).
    - `cli.py` – command-line interface for generating layer sets.
    - `generator.py` – RGB/depth ingestion, MiDaS estimation, layer stratification.
    - `synthesizer.py` – gaze-adaptive compositor and helpers (`GazeState`, `simulate_gaze_sequence`).
- `TS/` – optional virtual environment convenience folder (not required for reproduction).

```text
Through_the_Screen/
├─ README.md
├─ CONTEXT.md
├─ input/
├─ output/
├─ scripts/
│  └─ render_gaze_demo.py
├─ src/
│  └─ volumetric_layers/
│     ├─ __init__.py
│     ├─ cli.py
│     ├─ generator.py
│     └─ synthesizer.py
└─ TS/  (optional venv)
```

## Key Interfaces
- `encode_scene(frames, saliency_map) -> cpsl_layers`  
- `decode_cpsl(cpsl_layers, viewpoint) -> frame`  
- `evaluate_quality(ref, recon, roi_mask) -> {PSNR, SSIM, VMAF, ROI-PSNR}`  
- `measure_bitrate(cpsl_layers) -> Mbps`

## Known Issues / TODO
- Need to design an efficient ROI extraction and saliency map generation module (currently using DepthAnything V2 + semantic segmentation).  
- ROI-weighted metrics must be aligned with baselines to ensure fair comparison.  
- Further explore redundancy reduction across layers (e.g., layer fusion, residual coding).

## Insights
- maybe we should make the egde of the layers after the first layer transparent, and when reassemble them, we mix the RGB channels of the corresponding pixels.
