from pathlib import Path
import sys

import numpy as np

from qingxu.emotion_pipeline import (
    EmotionProbabilitySmoother,
    EMOTION_LABELS,
    preprocess_face,
    resolve_vit_model_path,
    select_primary_face,
)


def test_emotion_thread_exposes_seven_vit_labels():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "deidao"))
    try:
        from detector import EmotionDetectionThread
    finally:
        sys.path.remove(str(repo_root / "deidao"))

    thread = EmotionDetectionThread(model_path=resolve_vit_model_path(), source=0)

    assert list(thread._class_names_en) == list(EMOTION_LABELS)
    assert thread.get_class_names() == [
        "愤怒", "厌恶", "恐惧", "高兴", "平静", "悲伤", "惊讶"
    ]


def test_legacy_yolo_demo_falls_back_to_available_model(monkeypatch):
    qingxu_dir = Path(__file__).resolve().parents[1] / "qingxu"
    monkeypatch.chdir(qingxu_dir)
    sys.path.insert(0, str(qingxu_dir))
    try:
        import app as legacy_app
    finally:
        sys.path.remove(str(qingxu_dir))

    assert legacy_app._resolve_model_path({}, None).name == "best_yolo11s.onnx"


def test_preprocess_face_returns_normalized_chw_batch():
    frame = np.zeros((224, 224, 3), dtype=np.uint8)
    frame[:, :, 2] = 255  # BGR red channel

    pixel_values = preprocess_face(frame)

    assert pixel_values.shape == (1, 3, 224, 224)
    assert pixel_values.dtype == np.float32
    assert float(pixel_values.min()) >= -1.0
    assert float(pixel_values.max()) <= 1.0
    assert float(pixel_values[0, 0].mean()) > float(pixel_values[0, 2].mean())


def test_select_primary_face_prefers_largest_detection():
    faces = [
        (10, 10, 20, 20, 0.95),
        (50, 50, 80, 80, 0.70),
        (100, 100, 40, 40, 0.90),
    ]

    assert select_primary_face(faces) == faces[1]


def test_probability_smoother_uses_hysteresis_and_clears_after_missing_faces():
    labels = ["angry", "happy", "sad"]
    smoother = EmotionProbabilitySmoother(
        labels=labels,
        alpha=0.5,
        switch_margin=0.10,
        max_missing_updates=2,
    )

    happy = {"angry": 0.1, "happy": 0.8, "sad": 0.1}
    slight_sad = {"angry": 0.1, "happy": 0.49, "sad": 0.41}
    clear_sad = {"angry": 0.1, "happy": 0.1, "sad": 0.8}

    assert smoother.update(happy) == "happy"
    assert smoother.update(slight_sad) == "happy"
    assert smoother.update(clear_sad) == "sad"

    assert smoother.update(None) == "sad"
    assert smoother.update(None) == "sad"
    assert smoother.update(None) is None


def test_resolve_vit_model_path_finds_bundled_model():
    path = resolve_vit_model_path()

    assert isinstance(path, Path)
    assert path.is_file()
    assert path.name == "model.onnx"
