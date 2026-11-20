"""Offline evaluation for facial-emotion classifier backends."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Iterable, Sequence

import cv2
import numpy as np

from qingxu.emotion_backends import (
    CANONICAL_SEVEN_LABELS,
    EmotionClassifierBackend,
    normalize_emotion_label,
)
from qingxu.emotion_pipeline import MediaPipeFaceDetector, select_primary_face


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
OTHER_PREDICTION = "__other__"


@dataclass(frozen=True)
class BenchmarkSample:
    path: Path
    label: str
    media_type: str


def load_benchmark_samples(
    dataset_dir: str | Path,
    allowed_labels: Sequence[str] = CANONICAL_SEVEN_LABELS,
) -> list[BenchmarkSample]:
    """Load image/video samples from emotion-named folders.

    The default comparison is restricted to the seven labels supported by the
    current realtime UI. HSEmotion may still predict contempt; that prediction
    is counted as an error on the seven-label task.
    """
    root = Path(dataset_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Benchmark dataset directory not found: {root}")

    allowed = set(allowed_labels)
    samples = []
    skipped_labels = []
    for child in sorted(root.iterdir(), key=lambda path: path.name.lower()):
        if not child.is_dir():
            continue
        label = normalize_emotion_label(child.name)
        if label is None or label not in allowed:
            skipped_labels.append(child.name)
            continue
        for path in sorted(child.rglob("*"), key=lambda item: str(item).lower()):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix in IMAGE_EXTENSIONS:
                samples.append(BenchmarkSample(path, label, "image"))
            elif suffix in VIDEO_EXTENSIONS:
                samples.append(BenchmarkSample(path, label, "video"))
    if skipped_labels:
        print(f"[WARN] Skipped unsupported label folders: {', '.join(skipped_labels)}")
    if not samples:
        raise ValueError(f"No benchmark images/videos found under emotion folders in {root}")
    return samples


def _iter_frames(sample: BenchmarkSample, frame_step: int, max_frames_per_video: int | None):
    if sample.media_type == "image":
        # np.fromfile + imdecode handles Chinese folder names reliably on Windows.
        data = np.fromfile(str(sample.path), dtype=np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
        if frame is None:
            raise ValueError(f"Could not read image: {sample.path}")
        yield 0, frame
        return

    capture = cv2.VideoCapture(str(sample.path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {sample.path}")
    emitted = 0
    frame_index = -1
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            frame_index += 1
            if frame_index % frame_step:
                continue
            yield frame_index, frame
            emitted += 1
            if max_frames_per_video is not None and emitted >= max_frames_per_video:
                break
    finally:
        capture.release()


def _crop_primary_face(
    frame: np.ndarray,
    detector,
    margin: float,
    min_face_size: int,
) -> np.ndarray | None:
    primary = select_primary_face(detector.detect(frame))
    if primary is None:
        return None
    frame_height, frame_width = frame.shape[:2]
    x, y, width, height, _ = primary
    if width < min_face_size or height < min_face_size:
        return None
    margin_x = int(width * margin)
    margin_y = int(height * margin)
    left = max(0, x - margin_x)
    top = max(0, y - margin_y)
    right = min(frame_width, x + width + margin_x)
    bottom = min(frame_height, y + height + margin_y)
    if right - left < min_face_size or bottom - top < min_face_size:
        return None
    return frame[top:bottom, left:right]


def compute_metrics(
    truth: Sequence[str],
    predictions: Sequence[str],
    labels: Sequence[str],
    latencies_ms: Sequence[float],
) -> dict:
    if len(truth) != len(predictions) or len(truth) != len(latencies_ms):
        raise ValueError("truth, predictions, and latency sequences must have equal lengths")

    labels = tuple(labels)
    label_set = set(labels)
    prediction_columns = labels + (OTHER_PREDICTION,)
    confusion = {label: {predicted: 0 for predicted in prediction_columns} for label in labels}
    per_class = {
        label: {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}
        for label in labels
    }
    correct = 0
    for actual, predicted in zip(truth, predictions):
        if actual not in label_set:
            raise ValueError(f"Ground-truth label outside configured labels: {actual}")
        predicted_column = predicted if predicted in label_set else OTHER_PREDICTION
        confusion[actual][predicted_column] += 1
        per_class[actual]["support"] += 1
        if actual == predicted:
            correct += 1

    f1_values = []
    for label in labels:
        true_positive = confusion[label][label]
        false_positive = sum(confusion[other][label] for other in labels if other != label)
        false_negative = sum(confusion[label][other] for other in prediction_columns if other != label)
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label].update(
            precision=round(precision, 6),
            recall=round(recall, 6),
            f1=round(f1, 6),
        )
        f1_values.append(f1)

    evaluated = len(truth)
    return {
        "evaluated": evaluated,
        "correct": correct,
        "accuracy": round(correct / evaluated, 6) if evaluated else 0.0,
        "macro_f1": round(sum(f1_values) / len(f1_values), 6) if f1_values else 0.0,
        "average_latency_ms": round(float(np.mean(latencies_ms)), 3) if latencies_ms else 0.0,
        "per_class": per_class,
        "confusion": confusion,
    }


def evaluate_backends(
    samples: Sequence[BenchmarkSample],
    backends: Sequence[EmotionClassifierBackend],
    detector=None,
    frame_step: int = 5,
    max_frames_per_video: int | None = 20,
    face_margin: float = 0.18,
    min_face_size: int = 48,
    max_preview_errors: int = 20,
) -> dict:
    """Evaluate every backend on exactly the same detected face crop."""
    if frame_step < 1:
        raise ValueError("frame_step must be at least 1")
    if not backends:
        raise ValueError("At least one backend is required")

    # Trigger CUDA graph/JIT initialization before latency measurement.
    warmup_face = np.full((224, 224, 3), 127, dtype=np.uint8)
    for backend in backends:
        backend.predict(warmup_face)

    owns_detector = detector is None
    active_detector = detector or MediaPipeFaceDetector()
    results = {
        backend.name: {"truth": [], "predictions": [], "latencies_ms": []}
        for backend in backends
    }
    preview_errors = []
    read_errors = []
    no_face_count = 0
    evaluated_frames = 0

    try:
        for sample in samples:
            try:
                frames = list(_iter_frames(sample, frame_step, max_frames_per_video))
            except Exception as exc:
                read_errors.append({"sample": str(sample.path), "error": str(exc)})
                continue

            for frame_index, frame in frames:
                try:
                    face = _crop_primary_face(frame, active_detector, face_margin, min_face_size)
                except Exception as exc:
                    read_errors.append(
                        {
                            "sample": str(sample.path),
                            "frame": frame_index,
                            "error": f"face detection failed: {exc}",
                        }
                    )
                    continue
                if face is None:
                    no_face_count += 1
                    continue

                evaluated_frames += 1
                predictions = {}
                for backend in backends:
                    started = time.perf_counter()
                    probabilities = backend.predict(face)
                    latency_ms = (time.perf_counter() - started) * 1000.0
                    predicted = max(probabilities, key=probabilities.get)
                    results[backend.name]["truth"].append(sample.label)
                    results[backend.name]["predictions"].append(predicted)
                    results[backend.name]["latencies_ms"].append(latency_ms)
                    predictions[backend.name] = predicted

                incorrect = [name for name, predicted in predictions.items() if predicted != sample.label]
                if incorrect and len(preview_errors) < max_preview_errors:
                    preview_errors.append(
                        {
                            "sample": str(sample.path),
                            "frame": frame_index,
                            "truth": sample.label,
                            "predictions": predictions,
                        }
                    )
    finally:
        if owns_detector:
            active_detector.close()

    backend_reports = {}
    for backend in backends:
        result = results[backend.name]
        backend_reports[backend.name] = compute_metrics(
            truth=result["truth"],
            predictions=result["predictions"],
            labels=CANONICAL_SEVEN_LABELS,
            latencies_ms=result["latencies_ms"],
        )

    label_counts = Counter(sample.label for sample in samples)
    return {
        "dataset": {
            "total_samples": len(samples),
            "sample_labels": {label: label_counts.get(label, 0) for label in CANONICAL_SEVEN_LABELS},
            "evaluated_frames": evaluated_frames,
            "no_face_frames": no_face_count,
            "read_errors": read_errors,
        },
        "backends": backend_reports,
        "preview_errors": preview_errors,
    }


def render_markdown_report(report: dict) -> str:
    lines = [
        "# Emotion Model Benchmark",
        "",
        f"- Samples: {report['dataset']['total_samples']}",
        f"- Evaluated face frames: {report['dataset']['evaluated_frames']}",
        f"- Frames without a usable face: {report['dataset']['no_face_frames']}",
        "",
        "| Backend | Accuracy | Macro F1 | Avg latency (ms) | Evaluated |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, metrics in report["backends"].items():
        lines.append(
            f"| {name} | {metrics['accuracy']:.2%} | {metrics['macro_f1']:.3f} | "
            f"{metrics['average_latency_ms']:.1f} | {metrics['evaluated']} |"
        )

    lines.extend(["", "## Per-class F1", ""])
    backend_names = list(report["backends"])
    all_labels = list(CANONICAL_SEVEN_LABELS)
    lines.append("| Label | " + " | ".join(backend_names) + " |")
    lines.append("|---" * (len(backend_names) + 1) + "|")
    for label in all_labels:
        values = []
        for name in backend_names:
            metrics = report["backends"][name]["per_class"].get(label, {})
            values.append(f"{metrics.get('f1', 0.0):.3f}")
        lines.append(f"| {label} | " + " | ".join(values) + " |")

    lines.extend(["", "## Preview Errors", ""])
    if not report["preview_errors"]:
        lines.append("None in the retained preview window.")
    else:
        for error in report["preview_errors"]:
            predictions = ", ".join(f"{name}={value}" for name, value in error["predictions"].items())
            lines.append(f"- `{error['sample']}` frame {error['frame']}: truth={error['truth']}; {predictions}")
    return "\n".join(lines) + "\n"


def save_reports(report: dict, output_dir: str | Path, stem: str = "emotion_model_benchmark") -> tuple[Path, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / f"{stem}.json"
    markdown_path = output / f"{stem}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(render_markdown_report(report), encoding="utf-8")
    return json_path, markdown_path
