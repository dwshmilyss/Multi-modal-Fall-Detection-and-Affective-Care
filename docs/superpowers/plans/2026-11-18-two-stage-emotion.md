# Two-Stage Realtime Emotion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the popup's direct YOLO emotion inference with MediaPipe face detection plus ViT seven-class expression classification.

**Architecture:** Add a reusable `qingxu/emotion_pipeline.py` that owns face selection, preprocessing, ONNX inference, and temporal smoothing. Keep `EmotionDetectionThread` responsible only for camera lifecycle, drawing, and UI payload storage. Update model resolution and dialog labels to use the bundled ViT model by default.

**Tech Stack:** Python, OpenCV, MediaPipe Face Detection, ONNX Runtime, PyQt6 threads, pytest.

---

### Task 1: Pipeline Unit Tests

**Files:**

- Create: `test/test_emotion_pipeline.py`

- [x] Write tests for `preprocess_face`, `select_primary_face`, `EmotionProbabilitySmoother`, and `resolve_vit_model_path`.
- [x] Run `python -m pytest test/test_emotion_pipeline.py -q -s` and confirm all tests fail because the module does not exist.

### Task 2: Shared Two-Stage Pipeline

**Files:**

- Create: `qingxu/emotion_pipeline.py`

- [x] Implement labels, preprocessing, primary-face selection, probability smoothing, model path resolution, and `ViTFaceEmotionRecognizer`.
- [x] Run pipeline unit tests and confirm they pass.
- [x] Run a synthetic-frame recognizer smoke test without opening a camera.

### Task 3: Popup Thread Integration

**Files:**

- Modify: `deidao/detector.py`
- Modify: `deidao/main.py`
- Modify: `deidao/dialogs.py`

- [x] Replace YOLO loading and grayscale inference in `EmotionDetectionThread` with `ViTFaceEmotionRecognizer`.
- [x] Store smoothed confidence and seven-class probability distribution in the existing latest-frame payload.
- [x] Resolve `qingxu/pro_emotion/models/onnx/model.onnx` before legacy `best.onnx` candidates.
- [x] Update popup labels and initial seven-class rows.
- [x] Run emotion pipeline tests and available per-file project tests.

### Task 4: Verification and Documentation

**Files:**

- Modify: `qingxu/README.md`

- [x] Document the two-stage default, model files, runtime behavior, and legacy YOLO comparison path.
- [x] Run the focused test file and per-file project tests; aggregate pytest collection is blocked by the pre-existing `test_alarm_system.py` stdout wrapper closing pytest's capture stream.
- [x] Inspect the final diff and confirm no unrelated files changed.
