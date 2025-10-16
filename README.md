# Content-Promoted Scene Layers (CPSL) Toolkit

This repository hosts a research-grade pipeline for converting conventional
RGB/RGB-D media into **Content-Promoted Scene Layers**—explicit RGBA-depth
planes that preserve photorealistic detail while enabling lightweight,
view-dependent rendering. The project targets CVPR-style experiments where
visual quality and perceptual control matter more than hard real-time
constraints.

Key capabilities:
- Monocular depth estimation + stratification into a configurable number of
  semi-transparent layers (CPSL generator).
- Gaze-aware compositing of those layers to render novel viewpoints on the
  client (CPSL synthesizer).
- Batch pipelines for turning whole videos into per-frame layer sets and
  reassembled sequences.

## Requirements

- Python 3.10+
- PyTorch + torchvision (for MiDaS depth estimation)
- Pillow, NumPy
- OpenCV (`opencv-python`) for the video pipeline
- **Optional**: ffmpeg executable on PATH for MP4 export

Example environment setup:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision pillow numpy opencv-python
```

Install a GPU build of PyTorch if you want faster depth estimation. Add ffmpeg
via your package manager (`brew install ffmpeg` on macOS) if you plan to
produce MP4 outputs.

## Core Workflow

### 1. Generate CPSL layers from an image
```bash
python3 -m volumetric_layers.cli input/your_frame.png \
    --output output/layers_run \
    --num-layers 4 \
    --edge-softening 1.5 \
    --depth-smoothing 1.0 \
    --mask-morph-radius 1 \
    --save-composite
```
This writes `layer_XX` folders, a `metadata.json`, and (optionally)
`composite.png` under `output/layers_run/`.

### 2. Reassemble with simulated gaze
```bash
PYTHONPATH=src python3 scripts/render_gaze_demo.py output/layers_run \
    --output output/gaze_demo \
    --frames 7
```
Generates `frame_00.png … frame_06.png` in `output/gaze_demo/`. These represent
slight yaw/pitch/zoom changes around the capture pose.

### 3. Convert PNG sequence to video (optional)
```bash
python3 scripts/frames_to_video.py output/gaze_demo \
    --framerate 15 \
    --output output/gaze_demo.mp4
```
Requires ffmpeg.

### 4. Video → CPSL → reassembled frames
```bash
python3 scripts/video_to_cpsl.py input/demo.mp4 \
    --output output/demo_run \
    --num-layers 4 \
    --edge-softening 1.5 \
    --depth-smoothing 1.0 \
    --mask-morph-radius 1 \
    --write-video \
    --framerate 15
```
This produces:
- `output/demo_run/layers/frame_XXXX/` (per-frame CPSL assets)
- `output/demo_run/reassembled/frame_XXXX.png` (gaze-adaptive renders)
- Optionally `output/demo_run/reassembled.mp4`

## Repository Structure

```
Through_the_Screen/
├─ README.md
├─ CONTEXT.md                  # extended project brief / CVPR context
├─ input/                      # optional source images/videos
├─ output/                     # generated assets (ignored by git)
├─ scripts/
│  ├─ render_gaze_demo.py      # PNG sweep using simulated gaze
│  ├─ video_to_cpsl.py         # video → per-frame CPSL → renders
│  └─ frames_to_video.py       # stitch PNG frames into MP4 via ffmpeg
└─ src/
   └─ volumetric_layers/
      ├─ __init__.py            # package exports (Layer, MultiLayerGenerator, etc.)
      ├─ cli.py                 # CLI front-end for CPSL generation
      ├─ generator.py           # MiDaS depth + layer stratification & inpainting
      └─ synthesizer.py         # gaze-adaptive compositor / LayeredFrameComposer
```

## Key Modules
- `volumetric_layers.generator`
  - `MultiLayerGenerator`: wraps MiDaS depth estimation and converts each frame
    into layered RGBA + depth/alpha masks. Depth smoothing, mask morphology, and
    occlusion filling are configurable.
  - `LayeredFrameSet`: container of ordered `Layer` objects with statistics and
    a convenience composite method.
- `volumetric_layers.synthesizer`
  - `LayeredFrameComposer`: loads layer stacks (or accepts raw arrays) and
    reprojects them for a given `GazeState` (yaw, pitch, zoom).
  - `simulate_gaze_sequence`: utility to generate a short sweep of gaze poses.
- `volumetric_layers.cli`: command-line interface to run the generator on a
  single image.

## Usage Notes
- MiDaS is downloaded on first use via `torch.hub`. The default model (`DPT_Large`)
  favours quality; swap to `MiDaS_small` for faster processing.
- Layer masks are now refined with a lightweight semantic prior (default:
  `deeplabv3_mobilenet_v3_large`). Use `--segmentation-model` to switch backbones
  or `--no-semantics` to fall back to depth-only stratification.
- Layer ordering is far → near. Masks and depth arrays contain NaNs outside the
  layer support.
- Missing content in reassembled frames means the source frame never saw that
  region; capture multiple viewpoints or increase the layer budget for better
  coverage.
- When running the video pipeline, you can throttle processing with
  `--frame-step` and `--max-frames` to keep experiments lightweight.

Feel free to adapt the scripts for new datasets or plug the package into your
own experimentation harness. The toolkit is designed to be transparent and easy
to extend for perceptually driven volumetric video research.
