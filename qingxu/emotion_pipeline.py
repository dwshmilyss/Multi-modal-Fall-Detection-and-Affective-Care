"""Two-stage realtime face emotion recognition.

Stage one locates the primary face with MediaPipe. Stage two classifies the
aligned crop with the bundled ViT expression ONNX model. Keeping the stages
separate avoids asking one YOLO detector to learn both face localization and
subtle expression differences.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
import time

import cv2
import numpy as np


EMOTION_LABELS = ("angry", "disgust", "fear", "happy", "neutral", "sad", "surprise")
EMOTION_CN_MAP = {
    "angry": "愤怒",
    "disgust": "厌恶",
    "fear": "恐惧",
    "happy": "高兴",
    "neutral": "平静",
    "sad": "悲伤",
    "surprise": "惊讶",
}


def resolve_vit_model_path() -> Path:
    """Return the bundled ViT ONNX model path used by the realtime popup."""
    return Path(__file__).resolve().parent / "pro_emotion" / "models" / "onnx" / "model.onnx"


def preprocess_face(face_bgr: np.ndarray) -> np.ndarray:
    """Convert a BGR face crop to the normalized NCHW input used by ViT."""
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
    pixel_values = resized.astype(np.float32) / 255.0
    pixel_values = (pixel_values - 0.5) / 0.5
    pixel_values = np.transpose(pixel_values, (2, 0, 1))[None, :, :, :]
    return np.ascontiguousarray(pixel_values, dtype=np.float32)


def _face_fields(face) -> tuple[int, int, int, int, float] | None:
    if isinstance(face, dict):
        try:
            return (
                int(face["x"]),
                int(face["y"]),
                int(face["width"]),
                int(face["height"]),
                float(face.get("score", 0.0)),
            )
        except (KeyError, TypeError, ValueError):
            return None
    try:
        return (int(face[0]), int(face[1]), int(face[2]), int(face[3]), float(face[4]))
    except (IndexError, TypeError, ValueError):
        return None


def select_primary_face(faces: Iterable) -> tuple[int, int, int, int, float] | None:
    """Select the largest valid face, which is normally the nearest person."""
    primary = None
    primary_area = 0
    for face in faces:
        fields = _face_fields(face)
        if fields is None:
            continue
        x, y, width, height, score = fields
        if width <= 0 or height <= 0 or score <= 0.0:
            continue
        area = width * height
        if area > primary_area:
            primary = fields
            primary_area = area
    return primary


class EmotionProbabilitySmoother:
    """Smooth class probabilities and switch labels only with a clear margin."""

    def __init__(
        self,
        labels: Sequence[str],
        alpha: float = 0.35,
        switch_margin: float = 0.08,
        max_missing_updates: int = 2,
    ) -> None:
        if not labels:
            raise ValueError("labels must not be empty")
        if len(set(labels)) != len(labels):
            raise ValueError("labels must be unique")
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if switch_margin < 0.0:
            raise ValueError("switch_margin must be non-negative")
        if max_missing_updates < 0:
            raise ValueError("max_missing_updates must be non-negative")

        self.labels = tuple(labels)
        self.alpha = float(alpha)
        self.switch_margin = float(switch_margin)
        self.max_missing_updates = int(max_missing_updates)
        self._probabilities = {label: 0.0 for label in self.labels}
        self._current_label: str | None = None
        self._missing_updates = 0

    @property
    def probabilities(self) -> dict[str, float]:
        return {label: float(self._probabilities[label]) for label in self.labels}

    def update(self, probabilities: dict[str, float] | None) -> str | None:
        """Return the stable label after incorporating one inference result."""
        if probabilities is None:
            self._missing_updates += 1
            if self._missing_updates > self.max_missing_updates:
                self.reset()
            return self._current_label

        raw = np.asarray(
            [max(0.0, float(probabilities.get(label, 0.0))) for label in self.labels],
            dtype=np.float64,
        )
        total = float(raw.sum())
        if not np.isfinite(total) or total <= 0.0:
            self._missing_updates += 1
            if self._missing_updates > self.max_missing_updates:
                self.reset()
            return self._current_label
        raw /= total

        self._missing_updates = 0
        if self._current_label is None:
            self._probabilities = {label: float(value) for label, value in zip(self.labels, raw)}
            self._current_label = self.labels[int(np.argmax(raw))]
            return self._current_label

        alpha = self.alpha
        for label, value in zip(self.labels, raw):
            self._probabilities[label] = (1.0 - alpha) * self._probabilities[label] + alpha * float(value)

        top_label = max(self.labels, key=lambda label: self._probabilities[label])
        if top_label != self._current_label:
            if self._probabilities[top_label] > self._probabilities[self._current_label] + self.switch_margin:
                self._current_label = top_label
        return self._current_label

    def reset(self) -> None:
        self._probabilities = {label: 0.0 for label in self.labels}
        self._current_label = None
        self._missing_updates = 0


@dataclass(frozen=True)
class EmotionFrameResult:
    emotion_en: str | None
    emotion_cn: str | None
    confidence: float
    probabilities: dict[str, float]
    face_box: tuple[int, int, int, int] | None
    inference_ms: float


class MediaPipeFaceDetector:
    """Small wrapper that keeps MediaPipe resources in one place."""

    def __init__(self, min_detection_confidence: float = 0.5, model_selection: int = 1) -> None:
        import mediapipe as mp

        self._detection = mp.solutions.face_detection.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=min_detection_confidence,
        )

    def detect(self, frame_bgr: np.ndarray) -> list[tuple[int, int, int, int, float]]:
        height, width = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        output = self._detection.process(rgb)
        if not output.detections:
            return []

        faces = []
        for detection in output.detections:
            box = detection.location_data.relative_bounding_box
            x = max(0, int(box.xmin * width))
            y = max(0, int(box.ymin * height))
            width_px = min(width - x, int(box.width * width))
            height_px = min(height - y, int(box.height * height))
            faces.append((x, y, width_px, height_px, float(detection.score[0])))
        return faces

    def close(self) -> None:
        self._detection.close()


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = logits.astype(np.float64).flatten()
    values -= float(np.max(values))
    exponentials = np.exp(values)
    return exponentials / float(exponentials.sum())


class ViTFaceEmotionRecognizer:
    """MediaPipe face detection plus ViT seven-class expression classifier."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        min_detection_confidence: float = 0.5,
        face_margin: float = 0.18,
        min_face_size: int = 48,
        smoothing_alpha: float = 0.35,
        switch_margin: float = 0.08,
        max_missing_updates: int = 2,
    ) -> None:
        self.model_path = Path(model_path) if model_path is not None else resolve_vit_model_path()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"ViT emotion model not found: {self.model_path}")

        self.face_margin = float(face_margin)
        self.min_face_size = int(min_face_size)
        self.face_detector = MediaPipeFaceDetector(
            min_detection_confidence=min_detection_confidence,
            model_selection=1,
        )
        self.smoother = EmotionProbabilitySmoother(
            labels=EMOTION_LABELS,
            alpha=smoothing_alpha,
            switch_margin=switch_margin,
            max_missing_updates=max_missing_updates,
        )

        import onnxruntime as ort

        requested_providers = [
            provider for provider in ("CUDAExecutionProvider", "CPUExecutionProvider")
            if provider in ort.get_available_providers()
        ]
        self.providers = tuple(requested_providers or ["CPUExecutionProvider"])
        self.session = ort.InferenceSession(str(self.model_path), providers=list(self.providers))
        self.input_name = self.session.get_inputs()[0].name

    @property
    def provider_description(self) -> str:
        return ", ".join(self.session.get_providers()) or "unknown"

    def _crop_face(
        self, frame: np.ndarray, face: tuple[int, int, int, int, float]
    ) -> np.ndarray | None:
        height, width = frame.shape[:2]
        x, y, face_width, face_height, _ = face
        if face_width < self.min_face_size or face_height < self.min_face_size:
            return None

        margin_x = int(face_width * self.face_margin)
        margin_y = int(face_height * self.face_margin)
        left = max(0, x - margin_x)
        top = max(0, y - margin_y)
        right = min(width, x + face_width + margin_x)
        bottom = min(height, y + face_height + margin_y)
        if right - left < self.min_face_size or bottom - top < self.min_face_size:
            return None
        return frame[top:bottom, left:right]

    def _empty_result(self, inference_ms: float = 0.0) -> EmotionFrameResult:
        label = self.smoother.update(None)
        probabilities = self.smoother.probabilities
        confidence = probabilities.get(label or "", 0.0)
        return EmotionFrameResult(
            emotion_en=label,
            emotion_cn=EMOTION_CN_MAP.get(label or "", None),
            confidence=float(confidence),
            probabilities=probabilities,
            face_box=None,
            inference_ms=float(inference_ms),
        )

    def process(self, frame_bgr: np.ndarray) -> EmotionFrameResult:
        """Recognize one frame; classifier failure clears the current result."""
        started = time.perf_counter()
        if frame_bgr is None or not isinstance(frame_bgr, np.ndarray) or frame_bgr.size == 0:
            return self._empty_result()

        try:
            primary_face = select_primary_face(self.face_detector.detect(frame_bgr))
            if primary_face is None:
                return self._empty_result((time.perf_counter() - started) * 1000.0)

            face_crop = self._crop_face(frame_bgr, primary_face)
            if face_crop is None:
                return self._empty_result((time.perf_counter() - started) * 1000.0)

            pixel_values = preprocess_face(face_crop)
            logits = self.session.run(None, {self.input_name: pixel_values})[0]
            probabilities_array = _softmax(np.asarray(logits))
            raw_probabilities = {
                label: float(value) for label, value in zip(EMOTION_LABELS, probabilities_array)
            }
            label = self.smoother.update(raw_probabilities)
            smoothed = self.smoother.probabilities
            inference_ms = (time.perf_counter() - started) * 1000.0
            return EmotionFrameResult(
                emotion_en=label,
                emotion_cn=EMOTION_CN_MAP.get(label or "", None),
                confidence=float(smoothed.get(label or "", 0.0)),
                probabilities=smoothed,
                face_box=tuple(primary_face[:4]),
                inference_ms=inference_ms,
            )
        except Exception:
            return self._empty_result((time.perf_counter() - started) * 1000.0)

    def close(self) -> None:
        self.face_detector.close()

    def __enter__(self) -> "ViTFaceEmotionRecognizer":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def draw_emotion_result(frame_bgr: np.ndarray, result: EmotionFrameResult) -> np.ndarray:
    """Draw the primary face and emotion text with OpenCV-safe characters."""
    output = frame_bgr.copy()
    if result.face_box is not None:
        x, y, width, height = result.face_box
        cv2.rectangle(output, (x, y), (x + width, y + height), (56, 217, 169), 2)
        label = result.emotion_en or "no face"
        text = f"{label} {result.confidence * 100:.0f}%"
        cv2.putText(output, text, (x, max(18, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (56, 217, 169), 2)
    else:
        cv2.putText(
            output,
            "No face",
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (94, 163, 148),
            2,
        )
    return output
