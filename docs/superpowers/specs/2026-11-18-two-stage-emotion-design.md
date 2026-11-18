# Two-Stage Realtime Emotion Design

## Goal

Improve realtime camera emotion recognition by replacing the popup's direct YOLO face-emotion detector with a two-stage pipeline: face detection followed by a seven-class ViT expression classifier.

## Architecture

The shared pipeline lives in `qingxu/emotion_pipeline.py`. Each processed frame is converted to RGB, MediaPipe Face Detection locates candidate faces, the largest usable face is cropped with a small margin, and the crop is resized to 224x224 and normalized for the existing `trpakov/vit-face-expression` ONNX model. The classifier produces seven probabilities (`angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`).

A probability smoother uses exponential moving averages plus hysteresis. A different emotion only becomes the displayed emotion when its smoothed probability exceeds the current emotion by a configured margin. Missing faces are tolerated briefly while tracking recovers, then reset the output to `None`; stale labels are not shown after the face disappears.

`deidao.detector.EmotionDetectionThread` becomes the camera lifecycle and UI transport layer. It reads frames, calls the shared recognizer, draws the face box and current label, and stores the latest frame for the existing polling UI. It no longer converts frames to grayscale before inference. `deidao.main` resolves the bundled ViT ONNX by default and only falls back to legacy YOLO files.

## Error Handling

Missing ViT model, missing MediaPipe dependency, classifier inference exceptions, and invalid camera frames are logged to the existing dialog log without crashing the UI. Classifier failures clear the current emotion for that inference. Camera reopening behavior remains unchanged.

## Testing

Unit tests cover preprocessing shape/range, primary-face selection, probability smoothing and missing-face reset, and model path resolution. Existing project tests are rerun. A synthetic-image smoke test verifies that the full recognizer returns seven normalized probabilities without opening a camera.
