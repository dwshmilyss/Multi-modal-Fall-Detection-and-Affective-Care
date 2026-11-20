"""CLI for comparing ViT and HSEmotion expression classifiers."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qingxu.emotion_backends import create_benchmark_backends
from qingxu.emotion_benchmark import evaluate_backends, load_benchmark_samples, save_reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path, help="Root containing one folder per emotion label")
    parser.add_argument(
        "--backends",
        nargs="*",
        default=["vit", "hse_b0", "hse_b2"],
        choices=["vit", "hse_b0", "hse_b2"],
    )
    parser.add_argument("--frame-step", type=int, default=5)
    parser.add_argument("--max-frames-per-video", type=int, default=20)
    parser.add_argument("--min-face-size", type=int, default=48)
    parser.add_argument("--face-margin", type=float, default=0.18)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output" / "benchmarks",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = load_benchmark_samples(args.dataset_dir)
    backends = create_benchmark_backends(args.backends)
    report = evaluate_backends(
        samples=samples,
        backends=backends,
        frame_step=args.frame_step,
        max_frames_per_video=args.max_frames_per_video,
        face_margin=args.face_margin,
        min_face_size=args.min_face_size,
    )
    json_path, markdown_path = save_reports(report, args.output_dir)
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {markdown_path}")


if __name__ == "__main__":
    main()
