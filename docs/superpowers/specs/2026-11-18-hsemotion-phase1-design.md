# HSEmotion Phase 1 Design

## Goal

Add a local, reproducible comparison between the current ViT expression classifier and HSEmotion ONNX classifiers without changing the realtime popup default.

## Architecture

Add backend-agnostic classification in `qingxu/emotion_backends.py`. `ViTBackend` preserves the current 224x224 normalization and seven labels. `HSEmotionBackend` implements the HSEmotion preprocessing and supports the eight-class `enet_b0_8_best_vgaf` and `enet_b2_8` models.

Add `qingxu/emotion_benchmark.py` for dataset loading, same-face evaluation, metrics, and reports. A benchmark sample is a still image or a sampled video frame whose ground truth is represented by its parent emotion folder. MediaPipe finds the largest face once per selected frame; every backend classifies the identical crop. Reports include accuracy, macro F1, per-class precision/recall/F1, confusion counts, average latency, detected/evaluated counts, and preview errors.

`qingxu/benchmark_emotion_models.py` provides the CLI. `qingxu/download_hsemotion_models.py` restores ignored ONNX weights. The popup continues using `ViTFaceEmotionRecognizer` and its current default until measured results justify switching.

## Error Handling

Unknown labels are skipped and reported. Missing files, unreadable media, invalid frames, and missing faces are counted rather than aborting the benchmark. Backend initialization failures fail clearly before evaluation. Downloads write only to the ignored local model directory.

## Testing

Unit tests cover English/Chinese label normalization, HSEmotion preprocessing, backend creation, metrics, and same-face evaluation with deterministic fake detector/backend objects. A real-model smoke test verifies both HSEmotion files load and emit normalized probabilities.
