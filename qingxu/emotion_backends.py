"""Classifier adapters used by offline emotion-model comparison."""
from __future__ import annotations

from pathlib import Path
import os
import sys
from typing import Protocol

import cv2
import numpy as np

from qingxu.emotion_pipeline import resolve_vit_model_path


CANONICAL_SEVEN_LABELS = (
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
)
CANONICAL_EIGHT_LABELS = (
    "angry",
    "contempt",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
)

LABEL_ALIASES = {
    "angry": "angry",
    "anger": "angry",
    "愤怒": "angry",
    "disgust": "disgust",
    "厌恶": "disgust",
    "fear": "fear",
    "fearful": "fear",
    "恐惧": "fear",
    "happy": "happy",
    "happiness": "happy",
    "高兴": "happy",
    "开心": "happy",
    "neutral": "neutral",
    "calm": "neutral",
    "平静": "neutral",
    "中性": "neutral",
    "sad": "sad",
    "sadness": "sad",
    "悲伤": "sad",
    "surprise": "surprise",
    "surprised": "surprise",
    "惊讶": "surprise",
    "contempt": "contempt",
    "轻蔑": "contempt",
}

HSE_EIGHT_LABELS = (
    "angry",
    "contempt",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
)
HSE_SEVEN_LABELS = (
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
)


def normalize_emotion_label(value: str | None) -> str | None:
    """Normalize English variants and Chinese UI labels to canonical IDs."""
    if value is None:
        return None
    normalized = str(value).strip().lower()
    return LABEL_ALIASES.get(normalized)


def resolve_hsemotion_model_path(model_name: str) -> Path:
    """Return the local path used for an optional HSEmotion ONNX model."""
    supported = {"enet_b0_8_best_vgaf", "enet_b2_8"}
    if model_name not in supported:
        raise ValueError(f"Unsupported HSEmotion model: {model_name}; expected one of {sorted(supported)}")
    return (
        Path(__file__).resolve().parent
        / "models"
        / "affectnet_emotions"
        / "onnx"
        / f"{model_name}.onnx"
    )


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64).flatten()
    values -= float(np.max(values))
    exponentials = np.exp(values)
    return exponentials / float(exponentials.sum())


def _create_onnx_session(model_path: Path):
    # Loading torch's CUDA runtime first makes the ONNX Runtime CUDA provider
    # resolve the local cuDNN/cuBLAS DLLs reliably on Windows.
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.init()
    except Exception:
        pass

    import onnxruntime as ort

    ensure_onnxruntime_dll_paths()

    providers = [
        provider
        for provider in ("CUDAExecutionProvider", "CPUExecutionProvider")
        if provider in ort.get_available_providers()
    ]
    return ort.InferenceSession(str(model_path), providers=providers or ["CPUExecutionProvider"])


def ensure_onnxruntime_dll_paths(
    venv_root: Path | None = None,
    environment: dict[str, str] | None = None,
) -> list[str]:
    """Add local CUDA DLL directories used by onnxruntime-gpu."""
    root = Path(venv_root) if venv_root is not None else Path(sys.executable).parent.parent
    env = os.environ if environment is None else environment
    candidates = [
        root / "Lib" / "site-packages" / "nvidia" / "cudnn" / "bin",
        root / "Lib" / "site-packages" / "nvidia" / "cublas" / "bin",
        root / "Lib" / "site-packages" / "nvidia" / "cuda_nvrtc" / "bin",
        root / "Lib" / "site-packages" / "torch" / "lib",
    ]
    existing_path = env.get("PATH", "")
    existing_parts = {part.lower() for part in existing_path.split(os.pathsep) if part}
    added = []
    for candidate in candidates:
        candidate_text = str(candidate)
        if not candidate.is_dir() or candidate_text.lower() in existing_parts:
            continue
        added.append(candidate_text)
        existing_parts.add(candidate_text.lower())
    if added:
        env["PATH"] = os.pathsep.join(added + ([existing_path] if existing_path else []))
    return added


class EmotionClassifierBackend(Protocol):
    name: str
    labels: tuple[str, ...]

    def preprocess(self, face_bgr: np.ndarray) -> np.ndarray:
        ...

    def predict(self, face_bgr: np.ndarray) -> dict[str, float]:
        ...


class ViTBackend:
    """Current seven-class ViT model used by the realtime popup."""

    def __init__(self, model_path: str | Path | None = None, session=None) -> None:
        self.name = "vit"
        self.labels = CANONICAL_SEVEN_LABELS
        self.model_path = Path(model_path) if model_path is not None else resolve_vit_model_path()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"ViT model not found: {self.model_path}")
        self.session = session or _create_onnx_session(self.model_path)
        self.input_name = (
            self.session.get_inputs()[0].name
            if hasattr(self.session, "get_inputs")
            else "input"
        )

    def preprocess(self, face_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
        pixels = resized.astype(np.float32) / 255.0
        pixels = (pixels - 0.5) / 0.5
        pixels = np.transpose(pixels, (2, 0, 1))[None, :, :, :]
        return np.ascontiguousarray(pixels, dtype=np.float32)

    def predict(self, face_bgr: np.ndarray) -> dict[str, float]:
        logits = self.session.run(None, {self.input_name: self.preprocess(face_bgr)})[0]
        probabilities = _softmax(logits)
        return {label: float(value) for label, value in zip(self.labels, probabilities)}


class HSEmotionBackend:
    """HSEmotion EfficientNet ONNX adapter.

    HSEmotion's public ONNX implementation passes OpenCV BGR crops directly and
    applies ImageNet normalization. Preserve that behavior so this comparison
    measures the published model rather than a modified pipeline.
    """

    def __init__(self, model_name: str, model_path: str | Path | None = None, session=None) -> None:
        supported = {"enet_b0_8_best_vgaf", "enet_b2_8"}
        if model_name not in supported:
            raise ValueError(f"Unsupported HSEmotion model: {model_name}; expected one of {sorted(supported)}")
        self.name = "hse_b0" if model_name.startswith("enet_b0") else "hse_b2"
        self.model_name = model_name
        self.labels = HSE_EIGHT_LABELS
        self.input_size = 224 if model_name.startswith("enet_b0") else 260
        self.mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        self.model_path = Path(model_path) if model_path is not None else resolve_hsemotion_model_path(model_name)
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"HSEmotion model not found: {self.model_path}; "
                "run python qingxu/download_hsemotion_models.py"
            )
        self.session = session or _create_onnx_session(self.model_path)
        self.input_name = (
            self.session.get_inputs()[0].name
            if hasattr(self.session, "get_inputs")
            else "input"
        )

    def preprocess(self, face_bgr: np.ndarray) -> np.ndarray:
        resized = cv2.resize(face_bgr, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        pixels = resized.astype(np.float32) / 255.0
        pixels = (pixels - self.mean) / self.std
        pixels = np.transpose(pixels, (2, 0, 1))[None, :, :, :]
        return np.ascontiguousarray(pixels, dtype=np.float32)

    def predict(self, face_bgr: np.ndarray) -> dict[str, float]:
        logits = self.session.run(None, {self.input_name: self.preprocess(face_bgr)})[0][0]
        probabilities = _softmax(logits)
        return {label: float(value) for label, value in zip(self.labels, probabilities)}


def create_benchmark_backends(selected: list[str]) -> list[EmotionClassifierBackend]:
    """Create backend adapters from CLI-style backend names."""
    backends: list[EmotionClassifierBackend] = []
    for name in selected:
        if name == "vit":
            backends.append(ViTBackend())
        elif name in {"hse_b0", "enet_b0_8_best_vgaf"}:
            backends.append(HSEmotionBackend("enet_b0_8_best_vgaf"))
        elif name in {"hse_b2", "enet_b2_8"}:
            backends.append(HSEmotionBackend("enet_b2_8"))
        else:
            raise ValueError(f"Unknown benchmark backend: {name}")
    return backends
