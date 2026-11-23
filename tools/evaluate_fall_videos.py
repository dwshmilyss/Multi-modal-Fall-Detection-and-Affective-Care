"""Run the Qt-free URFD fall-video evaluation and write reports."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deidao.fall_metrics import (
    FallEvaluationReport,
    VideoEvaluationResult,
    load_manifest,
    render_markdown_report,
    summarize_results,
)
from deidao.fall_video_evaluator import FallVideoEvaluator


EvaluatorFactory = Callable[..., FallVideoEvaluator]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="test_videos/evaluation/urfd")
    parser.add_argument(
        "--manifest",
        default=None,
        help="manifest path; defaults to <dataset>/manifest.json",
    )
    parser.add_argument("--output", default="output/evaluation/fall")
    parser.add_argument("--model", default="model/yolov8n.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--split-rgb", action="store_true")
    parser.add_argument("--upscale", type=float, default=1.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--only-label", choices=("fall", "adl"))
    return parser.parse_args(argv)


def _validated_entries(manifest: dict) -> list[dict]:
    entries = manifest["entries"]
    for index, entry in enumerate(entries):
        label = entry.get("label")
        should_alarm = entry.get("should_alarm")
        activity_group = entry.get("activity_group")
        action_start = entry.get("action_start_seconds")
        max_delay = entry.get("max_alarm_delay_seconds", 2.0)

        if not isinstance(label, str) or not label:
            raise ValueError(f"manifest entry {index} must contain a label string")
        if not isinstance(should_alarm, bool):
            raise ValueError(f"manifest entry {index} must contain a should_alarm boolean")
        if not isinstance(activity_group, str) or not activity_group:
            raise ValueError(f"manifest entry {index} must contain an activity_group string")
        if action_start is not None and not isinstance(action_start, (int, float)):
            raise ValueError(f"manifest entry {index} has invalid action_start_seconds")
        if not isinstance(max_delay, (int, float)) or max_delay <= 0:
            raise ValueError(f"manifest entry {index} has invalid max_alarm_delay_seconds")

    return entries


def _write_reports(
    output_dir: Path,
    report: FallEvaluationReport,
    manifest: dict,
    results: Sequence[VideoEvaluationResult],
) -> None:
    payload = asdict(report)
    payload["dataset"] = manifest.get("dataset", report.dataset)
    payload["camera"] = manifest.get("camera", report.camera)
    if manifest.get("fps") is not None:
        payload["fps"] = manifest["fps"]
    payload["selected_count"] = len(results)
    payload["results"] = [asdict(result) for result in results]

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "fall_evaluation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "fall_evaluation.md").write_text(
        render_markdown_report(report),
        encoding="utf-8",
    )


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    evaluator_factory: EvaluatorFactory = FallVideoEvaluator,
) -> int:
    args = parse_args(argv)
    manifest_path = (
        Path(args.manifest)
        if args.manifest is not None
        else Path(args.dataset) / "manifest.json"
    )

    try:
        manifest = load_manifest(manifest_path)
        entries = _validated_entries(manifest)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"Invalid manifest {manifest_path}: {error}", file=sys.stderr)
        return 2

    if args.only_label is not None:
        entries = [entry for entry in entries if entry["label"] == args.only_label]
    if args.limit is not None:
        entries = entries[: max(0, args.limit)]
    if not entries:
        print("No manifest videos were selected for evaluation.", file=sys.stderr)
        return 1

    try:
        evaluator = evaluator_factory(
            model_path=args.model,
            device=args.device,
            split_rgb=args.split_rgb,
            upscale=args.upscale,
        )
    except Exception as error:
        print(f"Could not initialize fall-video evaluator: {error}", file=sys.stderr)
        return 2

    results = []
    try:
        for entry in entries:
            result = evaluator.evaluate_video(
                str(manifest_path.parent / entry["path"]),
                label=entry["label"],
                should_alarm=entry["should_alarm"],
                action_start_seconds=entry.get("action_start_seconds"),
                max_alarm_delay_seconds=entry.get("max_alarm_delay_seconds", 2.0),
                activity_group=entry["activity_group"],
            )
            results.append(result)
            if result.read_error is not None:
                print(f"Read failed for {result.path}: {result.read_error}", file=sys.stderr)
    except Exception as error:
        print(f"Evaluation failed: {error}", file=sys.stderr)
        return 1
    finally:
        try:
            evaluator.close()
        except Exception as cleanup_error:
            print(f"Evaluator cleanup failed: {cleanup_error}", file=sys.stderr)

    report = summarize_results(results)
    _write_reports(Path(args.output), report, manifest, results)
    print(f"Evaluated {report.evaluated_count} of {len(results)} selected videos.")
    print(f"Reports written to {Path(args.output).resolve()}")

    if not results or all(result.read_error is not None for result in results):
        print("All selected videos failed to read.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
