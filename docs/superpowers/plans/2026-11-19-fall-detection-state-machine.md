# Fall Detection State Machine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the current consensus bypass and replace per-frame heuristics with a tested, time-normalized, per-person fall state machine.

**Architecture:** Create a Qt-free `deidao/fall_engine.py` containing a lightweight person tracker and one single-person temporal engine. Refactor `DetectionThread` to feed timestamped pose observations into tracked engines and consume their state/alarm decisions. Keep camera, model, drawing, recording, database, and notification concerns in the existing thread.

**Tech Stack:** Python 3.8+, dataclasses, standard-library timing/collections, OpenCV/YOLO/MediaPipe integration, pytest.

---

### Task 1: Failing Engine Tests

**Files:**
- Create: `test/test_fall_engine.py`

- [x] Add tests for slow intentional lying, rapid fall confirmation and hold, rapid sitting, low-hip-only suppression, FPS-independent detection, and IoU tracker ID stability.
- [x] Run `.\.venv\Scripts\python.exe -m pytest test/test_fall_engine.py -q`.
- [x] Confirm the run fails because `deidao.fall_engine` does not exist.

### Task 2: Temporal Engine and Tracker

**Files:**
- Create: `deidao/fall_engine.py`

- [x] Implement `FallEngineSettings`, `FallDetectionEngine`, and `SimplePersonTracker`.
- [x] Use `time.monotonic`-compatible timestamps, elapsed-time velocities, standing-height normalization, stable sitting/lying recognition, weighted dynamic-plus-terminal consensus, alarm hold, and missing-pose decay.
- [x] Run `.\.venv\Scripts\python.exe -m pytest test/test_fall_engine.py -q` and confirm all focused tests pass.

### Task 3: Detector Integration

**Files:**
- Modify: `deidao/detector.py`

- [x] Collect YOLO person boxes each frame and assign stable IDs with `SimplePersonTracker`.
- [x] Maintain one `FallDetectionEngine` per ID and feed valid MediaPipe pose observations using actual frame timestamps.
- [x] Use engine state for labels and `alarm_active` for the UI; trigger recording once from `alarm_event`.
- [x] Mark low-visibility and missing pose observations so candidates decay instead of latching from stale evidence.
- [x] Run `.\.venv\Scripts\python.exe -m pytest test/test_fall_engine.py -q`.
- [x] Run `.\.venv\Scripts\python.exe -m py_compile deidao\fall_engine.py deidao\detector.py`.

### Task 4: Regression Verification and Documentation

**Files:**
- Modify: `md/FALL_DETECTION_OPTIMIZATION.md`
- Modify: `README.md`

- [x] Document the real-time thresholds, state machine, sitting/lying behavior, per-person tracking, and alarm hold.
- [x] Run focused fall-engine tests and available per-file tests. Full `pytest test -q -s` passes; normal capture mode remains incompatible with the pre-existing `test_alarm_system.py` stdout wrapper.
- [x] Inspect `git diff` and ensure no unrelated user changes were modified.

