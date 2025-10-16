"""Combine a sequence of synthesized frames into a single video file."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Stitch demo frames into a video")
    parser.add_argument("frames", type=Path, help="Directory containing frame_XX.png outputs")
    parser.add_argument(
        "--framerate",
        type=int,
        default=15,
        help="Playback frame rate for the resulting video",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/gaze_demo.mp4"),
        help="Destination video file",
    )
    parser.add_argument(
        "--ffmpeg",
        type=str,
        default="ffmpeg",
        help="ffmpeg executable name (override if using a custom path)",
    )
    args = parser.parse_args()

    if not args.frames.exists():
        raise FileNotFoundError(f"Frame directory not found: {args.frames}")

    pattern = str((args.frames / "frame_%02d.png").resolve())
    output = str(args.output.resolve())

    command = [
        args.ffmpeg,
        "-y",
        "-framerate",
        str(args.framerate),
        "-i",
        pattern,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        output,
    ]

    print("Running:", " ".join(command))
    subprocess.run(command, check=True)
    print("Video written to", output)


if __name__ == "__main__":
    main()
