"""Pure parsing and metrics for the UR Fall Detection Dataset.

This module deliberately stays independent of Qt, YOLO, OpenCV, and the
runtime detector so evaluation artifacts can be checked in unit tests.
"""

import csv
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Optional, Sequence, Tuple


URFD_SOURCE_URL = "https://fenix.ur.edu.pl/~mkepski/ds/uf.html"


@dataclass(frozen=True)
class URFDAnnotation:
    """First action and lying frames recorded by an official annotation file."""

    sequence: str
    first_falling_frame: Optional[int]
    first_lying_frame: Optional[int]
    has_low_posture: bool


@dataclass(frozen=True)
class VideoEvaluationResult:
    """Outcome of evaluating one manifest video."""

    path: str
    label: str
    should_alarm: bool
    alarm_times: Tuple[float, ...]
    action_start_seconds: Optional[float]
    max_alarm_delay_seconds: float
    activity_group: str
    frame_count: int
    fps: float
    detected_person_frame_count: int
    processing_fps: float
    features: Tuple[str, ...]
    read_error: Optional[str] = None


@dataclass(frozen=True)
class FallEvaluationReport:
    """Aggregated URFD evaluation evidence."""

    dataset: str
    camera: str
    evaluated_count: int
    read_errors: Tuple[str, ...]
    overall_metrics: Dict[str, Any]
    group_metrics: Dict[str, Dict[str, Any]]


def parse_urfd_annotation_csv(path: str | Path) -> Dict[str, URFDAnnotation]:
    """Parse official URFD rows, ignoring malformed lines.

    Only the first three fields are read: sequence, frame index, and official
    posture label. Labels 0 and 1 identify falling and lying postures.
    """

    annotations: Dict[str, URFDAnnotation] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as annotation_file:
        for row in csv.reader(annotation_file):
            if not row:
                continue
            try:
                sequence = row[0]
                frame = int(row[1])
                label = int(row[2])
            except (IndexError, ValueError):
                continue

            current = annotations.get(
                sequence,
                URFDAnnotation(sequence=sequence, first_falling_frame=None, first_lying_frame=None, has_low_posture=False),
            )
            first_falling_frame = current.first_falling_frame
            first_lying_frame = current.first_lying_frame
            if label == 0:
                first_falling_frame = frame if first_falling_frame is None else min(first_falling_frame, frame)
            elif label == 1:
                first_lying_frame = frame if first_lying_frame is None else min(first_lying_frame, frame)

            has_low_posture = current.has_low_posture or label in (0, 1)
            annotations[sequence] = URFDAnnotation(
                sequence=sequence,
                first_falling_frame=first_falling_frame,
                first_lying_frame=first_lying_frame,
                has_low_posture=has_low_posture,
            )
    return annotations


def build_urfd_manifest(
    dataset_dir: str | Path,
    falls_csv: str | Path,
    adls_csv: str | Path,
    fps: float = 30.0,
) -> dict:
    """Build a deterministic manifest for downloaded cam0 URFD videos."""

    if fps <= 0:
        raise ValueError("fps must be greater than zero")

    dataset_path = Path(dataset_dir)
    entries = []
    falls = parse_urfd_annotation_csv(falls_csv)
    adls = parse_urfd_annotation_csv(adls_csv)

    for label, annotations in (("fall", falls), ("adl", adls)):
        for video_path in sorted((dataset_path / label).glob("*-cam0.mp4")):
            if not video_path.is_file():
                continue
            sequence = video_path.stem.removesuffix("-cam0")
            annotation = annotations.get(sequence)

            entries.append(
                {
                    "path": PurePosixPath(label, video_path.name).as_posix(),
                    "label": label,
                    "should_alarm": label == "fall",
                    "action_start_seconds": (
                        annotation.first_falling_frame / fps
                        if annotation is not None and annotation.first_falling_frame is not None
                        else None
                    ),
                    "lying_start_seconds": (
                        annotation.first_lying_frame / fps
                        if annotation is not None and annotation.first_lying_frame is not None
                        else None
                    ),
                    "max_alarm_delay_seconds": 2.0,
                    "activity_group": "fall"
                    if label == "fall"
                    else "low_posture_adl"
                    if annotation is not None and annotation.has_low_posture
                    else "basic_adl",
                }
            )

    return {
        "dataset": "URFD",
        "camera": "cam0",
        "fps": float(fps),
        "source_url": URFD_SOURCE_URL,
        "entries": entries,
    }


def load_manifest(path: str | Path) -> dict:
    """Read and validate a manifest, resolving paths against its directory."""

    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)

    if not isinstance(manifest, dict) or not isinstance(manifest.get("entries"), list):
        raise ValueError(f"manifest must contain an entries list: {manifest_path}")

    for index, entry in enumerate(manifest["entries"]):
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ValueError(f"manifest entry {index} must contain a path string")
        video_path = manifest_path.parent / PurePosixPath(entry["path"])
        if not video_path.is_file():
            raise FileNotFoundError(video_path)

    return manifest


def _new_metrics() -> dict:
    return {
        "evaluated_count": 0,
        "tp": 0,
        "late_detected": 0,
        "fn": 0,
        "fp": 0,
        "tn": 0,
        "timely_alarm_count": 0,
        "timely_alarm_latency_total_seconds": 0.0,
    }


def _rate(numerator: int, denominator: int) -> Optional[float]:
    return numerator / denominator if denominator else None


def _finish_metrics(metrics: dict) -> dict:
    positive_count = metrics["tp"] + metrics["fn"]
    negative_count = metrics["fp"] + metrics["tn"]
    metrics["recall"] = _rate(metrics["tp"], positive_count)
    metrics["precision"] = _rate(metrics["tp"], metrics["tp"] + metrics["fp"])
    metrics["missed_alarm_rate"] = _rate(metrics["fn"], positive_count)
    metrics["fp_rate"] = _rate(metrics["fp"], negative_count)
    metrics["average_alarm_latency_seconds"] = (
        metrics["timely_alarm_latency_total_seconds"] / metrics["timely_alarm_count"]
        if metrics["timely_alarm_count"]
        else None
    )
    metrics.pop("timely_alarm_count", None)
    metrics.pop("timely_alarm_latency_total_seconds", None)
    return metrics


def summarize_results(results: Sequence[VideoEvaluationResult]) -> FallEvaluationReport:
    """Classify usable results and calculate overall and subgroup metrics.

    A late positive is a diagnostic count and is also included in the FN
    denominator. This keeps recall strictly a timely-detection metric.
    """

    metric_sets = {
        "overall": _new_metrics(),
        "fall": _new_metrics(),
        "all_adl": _new_metrics(),
        "basic_adl": _new_metrics(),
        "low_posture_adl": _new_metrics(),
    }
    read_errors = []

    for result in results:
        if result.read_error is not None:
            read_errors.append(f"{result.path}: {result.read_error}")
            continue

        group_names = ["overall", "fall" if result.should_alarm else "all_adl"]
        if not result.should_alarm and result.activity_group in metric_sets:
            group_names.append(result.activity_group)

        timely_latency = None
        if result.should_alarm:
            if result.action_start_seconds is not None:
                deadline = result.action_start_seconds + result.max_alarm_delay_seconds
                latencies = [
                    alarm_time - result.action_start_seconds
                    for alarm_time in result.alarm_times
                    if result.action_start_seconds <= alarm_time <= deadline
                ]
                if latencies:
                    timely_latency = min(latencies)

            classification = "tp" if timely_latency is not None else "fn"
            if timely_latency is None and result.alarm_times:
                metrics_late_increment = 1
            else:
                metrics_late_increment = 0
        else:
            classification = "fp" if result.alarm_times else "tn"
            metrics_late_increment = 0

        for group_name in group_names:
            metrics = metric_sets[group_name]
            metrics["evaluated_count"] += 1
            metrics[classification] += 1
            metrics["late_detected"] += metrics_late_increment
            if timely_latency is not None:
                metrics["timely_alarm_count"] += 1
                metrics["timely_alarm_latency_total_seconds"] += timely_latency

    group_metrics = {
        group_name: _finish_metrics(metrics)
        for group_name, metrics in metric_sets.items()
        if group_name != "overall"
    }
    overall_metrics = _finish_metrics(metric_sets["overall"])

    return FallEvaluationReport(
        dataset="URFD",
        camera="cam0",
        evaluated_count=overall_metrics["evaluated_count"],
        read_errors=tuple(read_errors),
        overall_metrics=overall_metrics,
        group_metrics=group_metrics,
    )


def _format_metric(value: Any, *, percent: bool = False, seconds: bool = False) -> str:
    if value is None:
        return "不适用"
    if percent:
        return f"{value:.2%}"
    if seconds:
        return f"{value:.3f} s"
    return str(value)


def render_markdown_report(report: FallEvaluationReport) -> str:
    """Render a concise Chinese report for algorithm-evidence review."""

    overall = report.overall_metrics
    lines = [
        "# URFD 跌倒检测评测报告",
        "",
        "数据集：UR Fall Detection Dataset",
        f"相机：{report.camera}",
        f"有效评测视频：{report.evaluated_count}",
        f"读取失败：{len(report.read_errors)}",
        "",
        "## 总体指标",
        "",
        f"- 跌倒样本：{overall['tp'] + overall['fn']}",
        f"- TP：{overall['tp']}",
        f"- late_detected：{overall['late_detected']}",
        f"- FN：{overall['fn']}",
        f"- FP：{overall['fp']}",
        f"- TN：{overall['tn']}",
        f"- 召回率：{_format_metric(overall['recall'], percent=True)}",
        f"- 精确率：{_format_metric(overall['precision'], percent=True)}",
        f"- 漏报率：{_format_metric(overall['missed_alarm_rate'], percent=True)}",
        f"- 误报率：{_format_metric(overall['fp_rate'], percent=True)}",
        f"- 平均及时报警延迟：{_format_metric(overall['average_alarm_latency_seconds'], seconds=True)}",
        "",
        "## 分组指标",
        "",
        "| 分组 | TP | late_detected | FN | FP | TN | 召回率 | 误报率 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for group_name in ("fall", "all_adl", "basic_adl", "low_posture_adl"):
        metrics = report.group_metrics[group_name]
        lines.append(
            f"| {group_name} | {metrics['tp']} | {metrics['late_detected']} | "
            f"{metrics['fn']} | {metrics['fp']} | {metrics['tn']} | "
            f"{_format_metric(metrics['recall'], percent=True)} | "
            f"{_format_metric(metrics['fp_rate'], percent=True)} |"
        )

    lines.extend(
        [
            "",
            "## 指标说明",
            "",
            "- 全量 ADL 是所有非跌倒日常活动样本，用于计算整体误报风险。",
            "- 基础 ADL 是官方日常活动中未出现 0/1 低姿态标注的样本。",
            "- 低姿态 ADL 是官方日常活动中出现 0/1 低姿态标注的样本，仍是负样本。",
            "- late_detected 不计入召回率；它只作为延迟诊断，召回率仅统计报警窗口内的及时检测。",
        ]
    )

    if report.read_errors:
        lines.extend(["", "## 读取失败", ""])
        lines.extend(f"- {error}" for error in report.read_errors)

    return "\n".join(lines) + "\n"
