"""Render a short gaze sweep using pre-generated layered frames."""
from __future__ import annotations

import argparse
from pathlib import Path

from volumetric_layers import LayeredFrameComposer, simulate_gaze_sequence


def main() -> None:
    parser = argparse.ArgumentParser(description="Render gaze-adaptive preview frames")
    parser.add_argument("layers", type=Path, help="Directory containing layered frame output")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/gaze_demo"),
        help="Directory to store synthesized views",
    )
    parser.add_argument("--frames", type=int, default=5, help="Number of gaze samples to render")
    args = parser.parse_args()

    composer = LayeredFrameComposer()
    gaze_sequence = simulate_gaze_sequence(num_frames=args.frames)

    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Rendering {len(gaze_sequence)} views from {args.layers} → {args.output}")
    for idx, gaze in enumerate(gaze_sequence):
        image = composer.compose_from_directory(args.layers, gaze)
        output_path = args.output / f"frame_{idx:02d}.png"
        image.save(output_path)
        print(f"  • wrote {output_path.name}")
    print("Done.")


if __name__ == "__main__":
    main()
