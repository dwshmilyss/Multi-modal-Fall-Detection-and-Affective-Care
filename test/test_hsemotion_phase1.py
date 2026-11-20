from pathlib import Path

import cv2
import numpy as np

from qingxu.emotion_backends import (
    CANONICAL_SEVEN_LABELS,
    HSEmotionBackend,
    ensure_onnxruntime_dll_paths,
    normalize_emotion_label,
    resolve_hsemotion_model_path,
)
from qingxu.emotion_benchmark import (
    compute_metrics,
    evaluate_backends,
    load_benchmark_samples,
    render_markdown_report,
    save_reports,
)
from qingxu.download_hsemotion_models import MODEL_METADATA, MODEL_NAMES, download_models


class FakeHSESession:
    def run(self, _outputs, feeds):
        pixels = feeds["input"]
        logits = np.zeros((1, 8), dtype=np.float32)
        logits[0, 4] = float(pixels.mean()) * 100.0
        return [logits]


def test_label_normalization_supports_english_and_chinese():
    assert normalize_emotion_label("Happiness") == "happy"
    assert normalize_emotion_label("Sadness") == "sad"
    assert normalize_emotion_label("惊讶") == "surprise"
    assert normalize_emotion_label("Contempt") == "contempt"
    assert normalize_emotion_label("unknown") is None


def test_hsemotion_backend_preprocesses_and_predicts_probabilities(tmp_path):
    model_path = tmp_path / "enet_b0_8_best_vgaf.onnx"
    model_path.write_bytes(b"fake")
    backend = HSEmotionBackend(
        model_name="enet_b0_8_best_vgaf",
        model_path=model_path,
        session=FakeHSESession(),
    )
    face = np.full((80, 80, 3), 127, dtype=np.uint8)

    pixels = backend.preprocess(face)
    probabilities = backend.predict(face)

    assert pixels.shape == (1, 3, 224, 224)
    assert pixels.dtype == np.float32
    assert probabilities["happy"] > 0.99
    assert abs(sum(probabilities.values()) - 1.0) < 1e-6


def test_hsemotion_model_path_uses_local_models_directory():
    path = resolve_hsemotion_model_path("enet_b2_8")

    assert path.name == "enet_b2_8.onnx"
    assert "qingxu" in path.parts
    assert "affectnet_emotions" in path.parts


def test_onnxruntime_dll_paths_are_added_once(tmp_path, monkeypatch):
    venv_root = tmp_path
    for relative in (
        "Lib/site-packages/nvidia/cudnn/bin",
        "Lib/site-packages/nvidia/cublas/bin",
        "Lib/site-packages/torch/lib",
    ):
        directory = venv_root / relative
        directory.mkdir(parents=True)
    environment = {"PATH": r"C:\Windows\System32"}

    added = ensure_onnxruntime_dll_paths(venv_root=venv_root, environment=environment)

    assert len(added) == 3
    assert environment["PATH"].count(str(venv_root / "Lib/site-packages/nvidia/cudnn/bin")) == 1


def test_model_download_replaces_incomplete_file(tmp_path, monkeypatch):
    target = tmp_path / "enet_b0_8_best_vgaf.onnx"
    target.write_bytes(b"partial")

    def fake_urlretrieve(_url, destination):
        Path(destination).write_bytes(b"x" * MODEL_METADATA["enet_b0_8_best_vgaf"]["size_bytes"])

    monkeypatch.setattr("qingxu.download_hsemotion_models.urllib.request.urlretrieve", fake_urlretrieve)

    paths = download_models(["enet_b0_8_best_vgaf"], tmp_path)

    assert paths == [target]
    assert target.stat().st_size == MODEL_METADATA["enet_b0_8_best_vgaf"]["size_bytes"]
    assert not list(tmp_path.glob("*.part"))


def test_download_cli_exposes_both_models():
    assert MODEL_NAMES == tuple(sorted(MODEL_METADATA))


def test_save_reports_writes_json_and_markdown(tmp_path):
    metrics = compute_metrics(
        truth=["happy"],
        predictions=["happy"],
        labels=CANONICAL_SEVEN_LABELS,
        latencies_ms=[12.5],
    )
    report = {
        "dataset": {
            "total_samples": 1,
            "evaluated_frames": 1,
            "no_face_frames": 0,
        },
        "backends": {"vit": metrics},
        "preview_errors": [],
    }

    json_path, markdown_path = save_reports(report, tmp_path)

    assert json_path.is_file()
    assert markdown_path.is_file()
    assert "Emotion Model Benchmark" in markdown_path.read_text(encoding="utf-8")


def test_model_download_retries_transient_network_failure(tmp_path, monkeypatch):
    calls = []

    def flaky_urlretrieve(_url, destination):
        calls.append(destination)
        if len(calls) == 1:
            raise OSError("proxy closed connection")
        Path(destination).write_bytes(b"x" * MODEL_METADATA["enet_b2_8"]["size_bytes"])

    monkeypatch.setattr(
        "qingxu.download_hsemotion_models.urllib.request.urlretrieve",
        flaky_urlretrieve,
    )

    paths = download_models(["enet_b2_8"], tmp_path, retry_delay_seconds=0)

    assert len(calls) == 2
    assert paths == [tmp_path / "enet_b2_8.onnx"]


def test_metrics_include_accuracy_macro_f1_and_confusion():
    labels = ["happy", "sad"]
    truth = ["happy", "happy", "sad", "sad", "sad"]
    predictions = ["happy", "sad", "sad", "happy", "sad"]

    metrics = compute_metrics(
        truth=truth,
        predictions=predictions,
        labels=labels,
        latencies_ms=[10.0, 11.0, 12.0, 13.0, 14.0],
    )

    assert metrics["evaluated"] == 5
    assert metrics["correct"] == 3
    assert metrics["accuracy"] == 0.6
    assert 0.0 < metrics["macro_f1"] <= 1.0
    assert metrics["average_latency_ms"] == 12.0
    assert metrics["confusion"]["happy"]["sad"] == 1
    assert metrics["confusion"]["sad"]["happy"] == 1


def test_metrics_treat_unsupported_prediction_as_error_without_breaking_macro_f1():
    metrics = compute_metrics(
        truth=["happy"],
        predictions=["contempt"],
        labels=["happy", "sad"],
        latencies_ms=[8.0],
    )

    assert metrics["evaluated"] == 1
    assert metrics["correct"] == 0
    assert metrics["accuracy"] == 0.0
    assert metrics["macro_f1"] == 0.0
    assert metrics["confusion"]["happy"]["__other__"] == 1


def test_benchmark_uses_identical_face_crop_for_all_backends(tmp_path):
    happy_dir = tmp_path / "happy"
    sad_dir = tmp_path / "悲伤"
    happy_dir.mkdir()
    sad_dir.mkdir()
    for directory, value, filename in (
        (happy_dir, 220, "happy.png"),
        (sad_dir, 30, "sad.png"),
    ):
        encoded, image = cv2.imencode(".png", np.full((100, 100, 3), value, np.uint8))
        assert encoded
        (directory / filename).write_bytes(image.tobytes())

    samples = load_benchmark_samples(tmp_path)

    class BrightnessBackend:
        def __init__(self, name, threshold):
            self.name = name
            self.threshold = threshold
            self.labels = CANONICAL_SEVEN_LABELS

        def predict(self, face):
            value = float(face.mean()) / 255.0
            return {"happy": value, "sad": 1.0 - value}

    class FixedFaceDetector:
        crops = []

        def detect(self, frame):
            return [(10, 10, 80, 80, 0.99)]

    backends = [
        BrightnessBackend("vit", 0.5),
        BrightnessBackend("hse_b0", 0.5),
    ]
    report = evaluate_backends(
        samples=samples,
        backends=backends,
        detector=FixedFaceDetector(),
        max_preview_errors=2,
    )

    assert len(samples) == 2
    assert report["dataset"]["total_samples"] == 2
    assert report["dataset"]["evaluated_frames"] == 2
    for backend_name in ("vit", "hse_b0"):
        assert report["backends"][backend_name]["evaluated"] == 2
        assert report["backends"][backend_name]["accuracy"] == 1.0
    assert render_markdown_report(report).startswith("# Emotion Model Benchmark")


def test_benchmark_warms_backend_before_measuring_latency(tmp_path):
    happy_dir = tmp_path / "happy"
    happy_dir.mkdir()
    encoded, image = cv2.imencode(".png", np.full((100, 100, 3), 220, np.uint8))
    assert encoded
    (happy_dir / "sample.png").write_bytes(image.tobytes())

    class CountingBackend:
        name = "counting"
        labels = CANONICAL_SEVEN_LABELS
        calls = []

        def predict(self, face):
            self.calls.append(face.shape)
            return {"happy": 1.0}

    class FixedFaceDetector:
        def detect(self, frame):
            return [(10, 10, 80, 80, 0.99)]

    backend = CountingBackend()
    report = evaluate_backends(
        samples=load_benchmark_samples(tmp_path),
        backends=[backend],
        detector=FixedFaceDetector(),
    )

    assert len(backend.calls) == 2
    assert backend.calls[0] == (224, 224, 3)
    assert report["backends"][backend.name]["evaluated"] == 1
