# HSEmotion Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compare the current ViT classifier against HSEmotion ONNX classifiers on the same local face samples.

**Architecture:** Put classifier adapters and label normalization in `qingxu/emotion_backends.py`. Put benchmark dataset loading, evaluation, metrics, and report serialization in `qingxu/emotion_benchmark.py`. Keep CLI wrappers thin and leave the realtime popup default unchanged.

**Tech Stack:** Python, OpenCV, MediaPipe, ONNX Runtime, pytest.

---

### Task 1: Failing Tests

**Files:**

- Create: `test/test_hsemotion_phase1.py`

- [x] Test Chinese/English label normalization, HSEmotion preprocessing, backend creation, metric calculation, and deterministic benchmark evaluation.
- [x] Run the focused test and verify it fails because the new modules do not exist.

### Task 2: Backends

**Files:**

- Create: `qingxu/emotion_backends.py`
- Modify: `test/test_hsemotion_phase1.py`

- [x] Implement canonical labels, normalization, ViT adapter, HSEmotion adapter, backend factory, and HSE model path resolution.
- [x] Run the focused test and verify backend tests pass.

### Task 3: Benchmark

**Files:**

- Create: `qingxu/emotion_benchmark.py`
- Create: `qingxu/benchmark_emotion_models.py`
- Modify: `test/test_hsemotion_phase1.py`

- [x] Implement image/video dataset loading, face cropping, same-crop evaluation, metrics, JSON/Markdown reports, and CLI options.
- [x] Run the focused test and verify benchmark tests pass.

### Task 4: Real Smoke and Verification

**Files:**

- Modify: `qingxu/README.md`
- Modify: `test/test_hsemotion_phase1.py`

- [x] Download and load the real `enet_b0_8_best_vgaf` ONNX model; it emits eight normalized probabilities and uses the CUDA provider.
- [x] Download and load the real `enet_b2_8` ONNX model; it emits eight normalized probabilities and uses the CUDA provider.
- [x] Run a fake-detector end-to-end benchmark.
- [x] Update usage documentation.
- [x] Run emotion tests, project tests, syntax compilation, and final diff checks.
