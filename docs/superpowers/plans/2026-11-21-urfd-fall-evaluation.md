# URFD 跌倒评测中心实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible offline evaluation pipeline for the temporal fall detector using the public UR Fall Detection Dataset.

**Architecture:** A pure-Python metrics layer parses URFD official frame annotations and computes sequence-level TP/FP/FN/TN plus alarm latency. A downloader fetches only official `cam0` videos and annotation CSVs, then writes an ignored local manifest. A Qt-free headless evaluator crops the right RGB half of each URFD frame and runs the same YOLO, MediaPipe observation construction, person tracker, and `FallDetectionEngine` decision path used by the desktop application.

**Tech Stack:** Python 3.11, OpenCV, Ultralytics YOLOv8, MediaPipe Pose, pytest, standard-library `urllib.request` and `json`.

---

## Scope

This plan implements only the URFD evidence-center milestone. It does not implement the risk engine, emergency controller, privacy mode, password migration, or demo preflight; those remain later milestones under the approved design.

## Repository Layout

- Create `deidao/fall_metrics.py`: annotation parsing, manifest handling, result classification, metrics, and Markdown rendering.
- Create `deidao/pose_observation.py`: shared MediaPipe-to-`FallDetectionEngine` observation construction.
- Modify `deidao/detector.py`: replace inline temporal-thread observation construction with the shared helper.
- Create `tools/download_urfd.py`: direct official downloader and manifest generator.
- Create `deidao/fall_video_evaluator.py`: Qt-free YOLO + MediaPipe + temporal engine evaluator.
- Create `tools/evaluate_fall_videos.py`: evaluator CLI.
- Create `tools/aggregate_evaluation.py`: combine fall and emotion reports.
- Create `test/test_fall_metrics.py`, `test/test_download_urfd.py`, `test/test_pose_observation.py`, `test/test_fall_video_evaluator.py`, and `test/test_aggregate_evaluation.py`.
- Modify `.gitignore` to exclude generated URFD media, manifests, CSVs, and evaluation reports.
- Modify `README.md` with download, evaluation, license, citation, and metric interpretation.

All production code follows red-green-refactor: each task adds a failing test, confirms the expected failure, implements the minimum behavior, confirms the focused test passes, and commits only that task's files.

### Task 1: URFD Annotations, Manifest, and Metrics

**Files:**

- Create: `deidao/fall_metrics.py`
- Create: `test/test_fall_metrics.py`
- Modify: `.gitignore`

- [x] **Step 1: Write failing annotation and manifest tests**

Create tests that cover:

1. `parse_urfd_annotation_csv("urfall-cam0-falls.csv")` groups rows by sequence, records the first `label == 0` frame as `first_falling_frame`, the first `label == 1` frame as `first_lying_frame`, and marks `has_low_posture`.
2. ADL annotations may also contain `0/1` low-posture frames; these remain negative ADL sequences but use `activity_group="low_posture_adl"`.
3. `build_urfd_manifest(dataset, falls_csv, adls_csv, fps=30.0)` emits relative paths, `fall/adl` labels, `should_alarm`, action start seconds, lying start seconds, and activity groups.
4. `load_manifest()` rejects any manifest entry whose video is missing.

Use this test fixture shape:

```python
def write_csv(path, rows):
    path.write_text(
        "\n".join(f"{sequence},{frame},{label},1,1,1,1,1,1,1,1" for sequence, frame, label in rows),
        encoding="utf-8",
    )
```

Expected manifest entry:

```python
{
    "path": "fall/fall-01-cam0.mp4",
    "label": "fall",
    "should_alarm": True,
    "action_start_seconds": 10 / 30.0,
    "lying_start_seconds": 20 / 30.0,
    "max_alarm_delay_seconds": 2.0,
    "activity_group": "fall",
}
```

- [x] **Step 2: Confirm Task 1 RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest test/test_fall_metrics.py -q
```

Expected failure: `ModuleNotFoundError: deidao.fall_metrics`.

- [x] **Step 3: Implement annotation and manifest parsing**

Create these public APIs:

| API | Contract |
|---|---|
| `URFDAnnotation` | Frozen dataclass with `sequence`, `first_falling_frame`, `first_lying_frame`, and `has_low_posture`. |
| `parse_urfd_annotation_csv(path: str | Path) -> dict[str, URFDAnnotation]` | Parse one official URFD annotation CSV into a sequence-keyed mapping. |
| `build_urfd_manifest(dataset_dir: str | Path, falls_csv: str | Path, adls_csv: str | Path, fps: float = 30.0) -> dict` | Build the complete manifest dictionary from downloaded videos and annotations. |
| `load_manifest(path: str | Path) -> dict` | Read and validate a manifest, raising `FileNotFoundError` for missing videos. |

Implementation details:

- Parse only the first three CSV fields: sequence, frame, label.
- Ignore malformed rows rather than aborting the file.
- Convert frame indexes to seconds using the supplied FPS.
- Fall sequences use `max_alarm_delay_seconds=2.0`.
- ADL sequences always use `should_alarm=False`.
- Basic ADL uses `activity_group="basic_adl"`.
- Any ADL containing official label `0` or `1` uses `activity_group="low_posture_adl"`.
- Include dataset metadata: `dataset="URFD"`, `camera="cam0"`, FPS, and official source URL.

- [x] **Step 4: Confirm Task 1 manifest tests pass**

Run the focused command again. Expected: all annotation and manifest tests pass.

- [x] **Step 5: Write failing result, metrics, and report tests**

Define expected result scenarios:

- One timely fall: action starts at 10.0 s, alarm at 10.8 s, allowed delay 2.0 s.
- One late fall: action starts at 10.0 s, alarm at 14.0 s.
- One missed fall: no alarm.
- One false-positive ADL.
- Two true-negative ADLs.
- Separate basic and low-posture ADL groups.

Required assertions:

```python
assert report["overall"]["tp"] == 1
assert report["overall"]["late_detected"] == 1
assert report["overall"]["fn"] == 2
assert report["overall"]["fp"] == 1
assert report["overall"]["tn"] == 2
assert report["overall"]["recall"] == pytest.approx(1 / 3)
assert report["overall"]["precision"] == pytest.approx(0.5)
assert report["overall"]["average_alarm_latency_seconds"] == pytest.approx(0.8)
assert report["groups"]["basic_adl"]["fp_rate"] == pytest.approx(0.5)
assert report["groups"]["low_posture_adl"]["fp_rate"] == pytest.approx(0.0)
```

Also test that Markdown contains:

- `# URFD 跌倒检测评测报告`
- `UR Fall Detection Dataset`
- fall sample/TP/late/FN counts
- 全量 ADL、基础 ADL、低姿态 ADL explanations
- explicit statement that `late_detected` is not counted as recall.

- [x] **Step 6: Confirm metrics/report RED**

Run the focused test command. Expected failure: missing `VideoEvaluationResult`, `FallEvaluationReport`, `summarize_results`, or `render_markdown_report`.

- [x] **Step 7: Implement result model, classification, metrics, and report**

Create the frozen dataclasses `VideoEvaluationResult` and `FallEvaluationReport` with exactly the fields listed in the tests. `VideoEvaluationResult` contains path, label, expected-alarm flag, alarm-time tuple, action start, maximum delay, activity group, frame count, FPS, detected-person-frame count, processing FPS, feature tuple, and optional read error. `FallEvaluationReport` contains dataset, camera, evaluated count, read errors, overall metrics, and group metrics.

Also create:

| API | Contract |
|---|---|
| `summarize_results(results: Sequence[VideoEvaluationResult]) -> dict` | Classify every usable result and calculate overall plus subgroup metrics. |
| `render_markdown_report(report: FallEvaluationReport) -> str` | Render the judge-facing Chinese Markdown report. |

Classification:

- `read_error` -> excluded from metrics and listed separately.
- Positive with an alarm in `[action_start, action_start + max_delay]` -> `tp`.
- Positive with only a later alarm -> `late_detected`.
- Positive with no alarm -> `fn`.
- Negative with any alarm -> `fp`.
- Negative without alarm -> `tn`.

For confusion-matrix metrics, both `late_detected` and no-alarm positives count as not timely and are therefore included in the FN denominator. `late_detected` remains a separate diagnostic count so it is never presented as recall.

Groups:

- `fall`
- `all_adl`
- `basic_adl`
- `low_posture_adl`

Each group reports evaluated count, TP, late, FN, FP, TN, recall, precision, missed-alarm rate, false-alarm rate, and average timely alarm latency when available.

- [x] **Step 8: Confirm Task 1 GREEN**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest test/test_fall_metrics.py -q
```

Expected: all Task 1 tests pass.

- [x] **Step 9: Ignore generated data**

Append to `.gitignore`:

```gitignore
test_videos/evaluation/
output/evaluation/
urfall-cam0-falls.csv
urfall-cam0-adls.csv
```

- [x] **Step 10: Commit Task 1**

```powershell
git add deidao/fall_metrics.py test/test_fall_metrics.py .gitignore
git commit -m "feat: add URFD fall evaluation metrics"
```

### Task 2: Official URFD Downloader

**Files:**

- Create: `tools/download_urfd.py`
- Create: `test/test_download_urfd.py`

- [x] **Step 1: Write failing downloader tests**

Cover:

1. `urfd_file_url("fall-01-cam0.mp4")` points to `https://fenix.ur.edu.pl/~mkepski/ds/data/fall-01-cam0.mp4`.
2. `build_download_plan()` contains exactly 70 `cam0` videos plus both official annotation CSVs.
3. Existing non-empty files are skipped.
4. A transient download error is retried.
5. A failed item is recorded and other downloads continue.
6. A manifest is generated after successful downloads.

- [x] **Step 2: Confirm downloader RED**

```powershell
.\.venv\Scripts\python.exe -m pytest test/test_download_urfd.py -q
```

Expected: `ModuleNotFoundError: tools.download_urfd`.

- [x] **Step 3: Implement downloader**

Public APIs:

| API | Contract |
|---|---|
| `URFD_BASE_URL` | Constant string `https://fenix.ur.edu.pl/~mkepski/ds/data/`. |
| `DownloadItem` | Frozen dataclass with `remote_name` and absolute `local_path`. |
| `urfd_file_url(remote_name: str) -> str` | Concatenate the official base URL and remote file name. |
| `build_download_plan(output_dir: str | Path) -> list[DownloadItem]` | Build the two-CSV plus 70-video plan. |
| `download_urfd(output_dir: str | Path, retries: int = 3, retry_delay_seconds: float = 2.0, create_manifest: bool = True) -> dict` | Download with retries, skip complete files, record failures, and optionally write the manifest. |

Implementation requirements:

- Download official `urfall-cam0-falls.csv` and `urfall-cam0-adls.csv`.
- Download `fall-01-cam0.mp4` through `fall-30-cam0.mp4`.
- Download `adl-01-cam0.mp4` through `adl-40-cam0.mp4`.
- Save videos under `fall/` and `adl/`.
- Use `.part` files and rename only after a complete response.
- Remove an incomplete `.part` after the final failed retry.
- Retry `OSError` and `urllib.error.URLError`.
- Return `{downloaded, skipped, failed, manifest}`.
- Build `manifest.json` with `build_urfd_manifest`.
- CLI defaults to `test_videos/evaluation/urfd` and supports `--retries`, `--retry-delay-seconds`, and `--no-manifest`.

- [x] **Step 4: Confirm downloader GREEN**

Run the focused downloader tests. Expected: all pass without network access; tests monkeypatch `_download_file`.

- [x] **Step 5: Commit Task 2**

```powershell
git add tools/download_urfd.py test/test_download_urfd.py
git commit -m "feat: add URFD dataset downloader"
```

### Task 3: Shared Pose Observation Builder

**Files:**

- Create: `deidao/pose_observation.py`
- Modify: `deidao/detector.py`
- Create: `test/test_pose_observation.py`

- [x] **Step 1: Write failing observation tests**

Create fake landmarks with `.x`, `.y`, and `.visibility`. Test:

1. A valid pose becomes a temporal-engine observation with `timestamp`, `torso_angle`, full-frame `hip_y` and `nose_y`, `bbox`, and average visibility.
2. A zero-size or invalid box returns `None`.
3. Torso angle stays in `[0, 90]`.

- [x] **Step 2: Confirm pose builder RED**

```powershell
.\.venv\Scripts\python.exe -m pytest test/test_pose_observation.py -q
```

Expected: `ModuleNotFoundError: deidao.pose_observation`.

- [x] **Step 3: Implement `build_pose_observation`**

Function contract:

`build_pose_observation(landmarks, nose_index, left_shoulder_index, right_shoulder_index, left_hip_index, right_hip_index, box, timestamp)` is keyword-only except for no positional arguments, accepts a landmark sequence, five integer landmark indexes, an `(x1, y1, x2, y2)` box, and a timestamp, then returns an observation dictionary or `None` for an invalid box.

This helper must not import MediaPipe, Qt, YOLO, or OpenCV. It uses only standard-library math and accepts landmark indexes, allowing both the app and evaluator to use the exact same calculation.

- [x] **Step 4: Confirm pose builder GREEN**

Run `test/test_pose_observation.py`. Expected: focused tests pass.

- [x] **Step 5: Refactor `TemporalDetectionThread`**

Replace the inline shoulder/hip/angle/visibility/dictionary block in the temporal thread with a call to `build_pose_observation`. Keep drawing, missing-pose behavior, tracker behavior, alarm handling, and Qt signals unchanged.

- [x] **Step 6: Run detector regression tests**

```powershell
.\.venv\Scripts\python.exe -m pytest test/test_pose_observation.py test/test_fall_engine.py -q
```

Expected: all tests pass.

- [x] **Step 7: Commit Task 3**

```powershell
git add deidao/pose_observation.py deidao/detector.py test/test_pose_observation.py
git commit -m "refactor: share temporal pose observation builder"
```

### Task 4: Qt-Free URFD Video Evaluator

**Files:**

- Create: `deidao/fall_video_evaluator.py`
- Create: `tools/evaluate_fall_videos.py`
- Create: `test/test_fall_video_evaluator.py`

- [x] **Step 1: Write failing evaluator tests**

Test pure functions:

1. `crop_urfd_rgb()` returns the exact right half of a `640x240` frame.
2. Timely alarm matching returns matched latency `0.2` for action start `3.0`, alarm `3.2`, and max delay `2.0`.
3. An alarm at `6.0` is late and returns latency `3.0`.

- [x] **Step 2: Confirm evaluator RED**

```powershell
.\.venv\Scripts\python.exe -m pytest test/test_fall_video_evaluator.py -q
```

Expected: `ModuleNotFoundError: deidao.fall_video_evaluator`.

- [x] **Step 3: Implement pure evaluator helpers**

Create:

| API | Contract |
|---|---|
| `crop_urfd_rgb(frame: np.ndarray) -> np.ndarray` | Return exactly the right half of a BGR frame without changing pixel values. |
| `evaluate_alarm_times(alarm_times: Sequence[float], action_start_seconds: float | None, max_alarm_delay_seconds: float) -> dict` | Return matched status, matched alarm time, and latency; a later alarm is not matched. |

- [x] **Step 4: Confirm helper tests GREEN**

Run the focused evaluator tests. Expected: pure helper tests pass.

- [x] **Step 5: Implement headless `FallVideoEvaluator`**

Class contract:

`FallVideoEvaluator(model_path="model/yolov8n.pt", device="auto", split_rgb=False, upscale=1.0)` owns one YOLO model and one MediaPipe Pose context. `close()` releases both. `evaluate_video(video_path, label, should_alarm, action_start_seconds, max_alarm_delay_seconds, activity_group)` evaluates one video and returns `VideoEvaluationResult`; all decision parameters after `video_path` are keyword-only.

Implementation requirements:

- Resolve `device="auto"` to CUDA when available, otherwise CPU.
- Load YOLO once per evaluator, not per video.
- Use one MediaPipe Pose context for all videos.
- Open and release one `cv2.VideoCapture` per video.
- Use `crop_urfd_rgb()` when `split_rgb=True`.
- Optionally resize after cropping with `upscale`.
- Use video time `frame_index / fps`, never wall time.
- Select YOLO COCO person class 0.
- Apply the same box clamping and minimum 10-pixel width/height checks as the temporal thread.
- Use `SimplePersonTracker`.
- Maintain one `FallDetectionEngine` per track ID.
- Use `build_pose_observation()`.
- Record `alarm_event` timestamps and features; do not count the three-second `alarm_active` hold as repeated alarms.
- Track detected-person frames and processing FPS.
- Return unreadable media as a `VideoEvaluationResult` whose `read_error` field contains the exception message instead of crashing the run.
- Close MediaPipe and capture in `finally`.

- [x] **Step 6: Implement evaluator CLI**

Command:

```powershell
.\.venv\Scripts\python.exe tools\evaluate_fall_videos.py --dataset test_videos\evaluation\urfd --split-rgb --upscale 2.0
```

Options:

- `--dataset`, default `test_videos/evaluation/urfd`
- `--manifest`, default `<dataset>/manifest.json`
- `--output`, default `output/evaluation/fall`
- `--model`, default `model/yolov8n.pt`
- `--device`, default `auto`
- `--split-rgb`
- `--upscale`, default `1.0`
- `--limit`
- `--only-label {fall,adl}`

Output:

- `fall_evaluation.json`
- `fall_evaluation.md`

Exit nonzero if the manifest is missing or every video failed.

- [x] **Step 7: Run evaluator and detector tests**

```powershell
.\.venv\Scripts\python.exe -m pytest test/test_fall_video_evaluator.py test/test_fall_engine.py -q
```

Expected: all pass.

- [x] **Step 8: Commit Task 4**

```powershell
git add deidao/fall_video_evaluator.py tools/evaluate_fall_videos.py test/test_fall_video_evaluator.py
git commit -m "feat: add offline fall video evaluator"
```

### Task 5: Evidence Aggregation and Documentation

**Files:**

- Create: `tools/aggregate_evaluation.py`
- Create: `test/test_aggregate_evaluation.py`
- Modify: `README.md`

- [x] **Step 1: Write failing aggregation test**

Use fake fall and emotion JSON reports. Assert aggregate Markdown contains:

- `# 算法评测证据汇总`
- URFD sample counts
- `28/30` fall detection summary
- `7.50%` all-ADL false-alarm rate
- emotion backend accuracy `72.00%`
- non-commercial license and citation
- non-medical statement

- [x] **Step 2: Confirm aggregator RED**

Expected module import failure.

- [x] **Step 3: Implement aggregator**

API:

| API | Contract |
|---|---|
| `load_json(path: str | Path) -> dict` | Read one UTF-8 JSON report. |
| `render_aggregate_markdown(fall_report_path: str | Path, emotion_report_path: str | Path | None = None) -> str` | Render the combined Chinese evidence Markdown; missing emotion data is omitted explicitly. |

CLI supports `--fall-report`, optional `--emotion-report`, and `--output`. Missing emotion data is explicitly omitted, never invented.

- [x] **Step 4: Confirm aggregator GREEN**

Run the aggregation test. Expected: pass.

- [x] **Step 5: Update README**

Add `算法评测证据中心（URFD）` with:

- Dataset source, composition, camera choice, and license.
- Required Kwolek/Kepski citation.
- Download command.
- Evaluation command.
- Aggregation command.
- Explanation that raw frames are depth left / RGB right and `--split-rgb` uses the right half.
- Timely versus late detection definitions.
- Basic ADL versus low-posture ADL.
- Non-medical statement.
- Generated media and reports are ignored.

- [x] **Step 6: Run all focused tests**

```powershell
.\.venv\Scripts\python.exe -m pytest test/test_fall_metrics.py test/test_download_urfd.py test/test_pose_observation.py test/test_fall_video_evaluator.py test/test_aggregate_evaluation.py -q
```

Expected: all focused tests pass.

- [x] **Step 7: Commit Task 5**

```powershell
git add tools/aggregate_evaluation.py test/test_aggregate_evaluation.py README.md
git commit -m "docs: document URFD fall evaluation pipeline"
```

### Task 6: Real URFD Smoke Run and Full Verification

**Files:**

- No intentional production changes unless a real defect is found.

- [ ] **Step 1: Download full URFD cam0 data**

```powershell
.\.venv\Scripts\python.exe tools\download_urfd.py --output test_videos\evaluation\urfd
```

Verify:

- 30 fall MP4s.
- 40 ADL MP4s.
- Both official annotation CSVs.
- 70-entry manifest.
- Generated files do not appear in Git status.

- [ ] **Step 2: Run two-video smoke evaluation**

```powershell
.\.venv\Scripts\python.exe tools\evaluate_fall_videos.py --dataset test_videos\evaluation\urfd --output output\evaluation\fall-smoke --split-rgb --upscale 2.0 --limit 2
```

Verify JSON/Markdown creation, model loading, frame processing, tracker use, and event collection.

- [ ] **Step 3: Fix real defects through TDD**

For any smoke defect:

1. Add the smallest failing test.
2. Confirm the expected failure.
3. Implement the minimal fix.
4. Confirm focused pass.
5. Re-run smoke.

- [ ] **Step 4: Run all 70 videos**

```powershell
.\.venv\Scripts\python.exe tools\evaluate_fall_videos.py --dataset test_videos\evaluation\urfd --output output\evaluation\fall --split-rgb --upscale 2.0
```

Expected report groups: fall, all ADL, basic ADL, low-posture ADL. Late detections remain separate from recall.

- [ ] **Step 5: Aggregate emotion report if available**

Use `qingxu/output/benchmarks/emotion_model_benchmark.json` when it exists. Otherwise omit emotion metrics and state fall-only scope.

- [ ] **Step 6: Run full test suite**

```powershell
.\.venv\Scripts\python.exe -m pytest test -q -s
```

Expected exit code 0.

- [ ] **Step 7: Verify generated data is ignored**

```powershell
git status --short
```

Expected no URFD media, CSV, manifest, or evaluation output paths.

- [ ] **Step 8: Commit real-defect fixes only**

If Task 6 exposed a defect, inspect `git diff --name-only` and stage each modified path explicitly after review. Then run:

```powershell
git commit -m "fix: harden URFD evaluation pipeline"
```

If no production changes were needed, do not create an empty commit.

## Verification Summary

Implementation is complete only when:

- All focused tests pass.
- Full `pytest test -q -s` exits successfully.
- Downloader creates 30 fall videos, 40 ADL videos, two annotation CSVs, and a 70-video manifest.
- Two-video smoke evaluation succeeds.
- Full 70-video evaluation writes JSON and Markdown.
- Generated media and reports are ignored by Git.
- README documents source, license, citation, commands, and metric interpretation.
