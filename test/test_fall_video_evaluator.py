import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from deidao.fall_engine import FallDecision
from deidao.fall_metrics import VideoEvaluationResult
from deidao.fall_video_evaluator import (
    FallVideoEvaluator,
    crop_urfd_rgb,
    evaluate_alarm_times,
)


def load_cli_module():
    module_path = Path(__file__).resolve().parents[1] / "tools" / "evaluate_fall_videos.py"
    spec = importlib.util.spec_from_file_location("evaluate_fall_videos_cli", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeBoxes:
    def __init__(self, xyxy, cls):
        self.xyxy = xyxy
        self.cls = cls


class FakeDetectionResult:
    def __init__(self, boxes):
        self.boxes = boxes


class FakeModel:
    def __init__(self, boxes_by_frame):
        self.boxes_by_frame = boxes_by_frame
        self.frames = []
        self.devices = []
        self.verbose_values = []
        self.closed = False

    def __call__(self, frame, device=None, verbose=False):
        self.frames.append(frame.copy())
        self.devices.append(device)
        self.verbose_values.append(verbose)
        index = len(self.frames) - 1
        boxes = self.boxes_by_frame[min(index, len(self.boxes_by_frame) - 1)]
        xyxy = [tuple(box[:4]) for box in boxes]
        classes = [box[4] for box in boxes]
        return [FakeDetectionResult(FakeBoxes(xyxy, classes))]

    def close(self):
        self.closed = True


class FakeLandmark:
    def __init__(self, x, y, visibility=0.95):
        self.x = x
        self.y = y
        self.visibility = visibility


class FakePoseResult:
    def __init__(self, landmarks):
        self.pose_landmarks = (
            SimpleNamespace(landmark=landmarks) if landmarks is not None else None
        )


class FakePose:
    PoseLandmark = SimpleNamespace(
        NOSE=SimpleNamespace(value=0),
        LEFT_SHOULDER=SimpleNamespace(value=1),
        RIGHT_SHOULDER=SimpleNamespace(value=2),
        LEFT_HIP=SimpleNamespace(value=3),
        RIGHT_HIP=SimpleNamespace(value=4),
    )

    def __init__(self, landmarks_by_frame):
        self.landmarks_by_frame = landmarks_by_frame
        self.images = []
        self.closed = False

    def process(self, image):
        self.images.append(image.copy())
        index = len(self.images) - 1
        landmarks_index = min(index, len(self.landmarks_by_frame) - 1)
        return FakePoseResult(self.landmarks_by_frame[landmarks_index])

    def close(self):
        self.closed = True


class FakeCapture:
    def __init__(self, frames, fps=30.0, opened=True):
        self.frames = frames
        self.fps = fps
        self.opened = opened
        self.released = False
        self._index = 0

    def isOpened(self):
        return self.opened

    def get(self, prop):
        return self.fps if prop == cv2.CAP_PROP_FPS else 0.0

    def read(self):
        if self._index >= len(self.frames):
            return False, None
        frame = self.frames[self._index]
        self._index += 1
        return True, frame

    def release(self):
        self.released = True


class BrokenCapture:
    def __init__(self, error):
        self.error = error
        self.released = False

    def isOpened(self):
        raise self.error

    def get(self, prop):
        return 30.0

    def release(self):
        self.released = True


class FailingModel:
    def __call__(self, frame, device=None, verbose=False):
        raise RuntimeError("model inference failed")


class FailingPose:
    PoseLandmark = FakePose.PoseLandmark

    def process(self, image):
        raise RuntimeError("pose inference failed")

    def close(self):
        pass


class FailingReadCapture(FakeCapture):
    def read(self):
        raise RuntimeError("decoder failed")


class ReleaseFailureCapture(FakeCapture):
    def __init__(self, frames, fps=30.0):
        super().__init__(frames, fps=fps)
        self.release_failed = False

    def release(self):
        self.release_failed = True
        raise RuntimeError("release failed")


class CloseFailureResource:
    PoseLandmark = FakePose.PoseLandmark

    def __init__(self):
        self.close_attempted = False

    def close(self):
        self.close_attempted = True
        raise RuntimeError("close failed")


def landmarks_for_observation(box, *, angle, hip_y, nose_y):
    x1, y1, _, y2 = box
    height = max(1.0, y2 - y1)
    hip_relative = (hip_y - y1) / height
    nose_relative = (nose_y - y1) / height
    shoulder_relative = max(0.0, hip_relative - 0.18)
    horizontal_offset = (hip_relative - shoulder_relative) * np.tan(np.deg2rad(angle))

    return [
        FakeLandmark(0.50, nose_relative),
        FakeLandmark(0.50 - horizontal_offset / 2.0, shoulder_relative),
        FakeLandmark(0.50 + horizontal_offset / 2.0, shoulder_relative),
        FakeLandmark(0.50, hip_relative),
        FakeLandmark(0.50, hip_relative),
    ]


def make_fall_components(frame_count=30, fps=10):
    upright_box = (100.0, 100.0, 180.0, 450.0)
    boxes = []
    landmarks = []
    upright_frames = 12
    fall_frames = 8

    for index in range(frame_count):
        if index < upright_frames:
            box = upright_box
            observation = {"angle": 5.0, "hip_y": 220.0, "nose_y": 80.0}
        elif index < upright_frames + fall_frames:
            progress = (index - upright_frames + 1) / fall_frames
            box = (
                100.0,
                100.0,
                100.0 + 240.0 * progress,
                285.0,
            )
            observation = {
                "angle": 5.0 + progress * 75.0,
                "hip_y": 220.0 + progress * 45.0,
                "nose_y": 80.0 + progress * 90.0,
            }
        else:
            box = (100.0, 100.0, 340.0, 285.0)
            observation = {"angle": 80.0, "hip_y": 265.0, "nose_y": 170.0}

        boxes.append([(*box, 0)])
        landmarks.append(landmarks_for_observation(box, **observation))

    frames = [np.full((240, 640, 3), index % 256, dtype=np.uint8) for index in range(frame_count)]
    return FakeModel(boxes), FakePose(landmarks), FakeCapture(frames, fps=fps)


def make_evaluator(model, pose, capture, *, split_rgb=True, upscale=1.0, cuda=False):
    return FallVideoEvaluator(
        model_path="fake-model.pt",
        device="auto",
        split_rgb=split_rgb,
        upscale=upscale,
        _model=model,
        _pose=pose,
        _capture_factory=lambda video_path: capture,
        _torch_cuda_available=cuda,
    )


def test_crop_urfd_rgb_returns_exact_right_half_without_resampling():
    frame = np.arange(240 * 640 * 3, dtype=np.uint8).reshape(240, 640, 3)

    cropped = crop_urfd_rgb(frame)

    assert cropped.shape == (240, 320, 3)
    assert np.array_equal(cropped, frame[:, 320:])


def test_evaluate_alarm_times_matches_the_earliest_timely_alarm():
    match = evaluate_alarm_times((3.0, 3.2, 5.2), action_start_seconds=3.0, max_alarm_delay_seconds=2.0)

    assert match == {
        "status": "timely",
        "matched_alarm_time": 3.0,
        "latency_seconds": 0.0,
    }


def test_evaluate_alarm_times_reports_timely_latency():
    match = evaluate_alarm_times((3.2,), action_start_seconds=3.0, max_alarm_delay_seconds=2.0)

    assert match["status"] == "timely"
    assert match["matched_alarm_time"] == pytest.approx(3.2)
    assert match["latency_seconds"] == pytest.approx(0.2)


def test_evaluate_alarm_times_reports_a_post_deadline_alarm_as_late():
    match = evaluate_alarm_times((6.0,), action_start_seconds=3.0, max_alarm_delay_seconds=2.0)

    assert match == {
        "status": "late",
        "matched_alarm_time": 6.0,
        "latency_seconds": 3.0,
    }


def test_evaluate_alarm_times_is_undefined_without_an_action_start():
    match = evaluate_alarm_times((3.2,), action_start_seconds=None, max_alarm_delay_seconds=2.0)

    assert match == {
        "status": "undefined",
        "matched_alarm_time": None,
        "latency_seconds": None,
    }


def test_evaluate_video_uses_frame_time_and_collects_one_alarm_event(monkeypatch):
    model, pose, capture = make_fall_components()
    evaluator = make_evaluator(model, pose, capture, cuda=True)
    clock = iter([100.0, 101.0])
    monkeypatch.setattr("deidao.fall_video_evaluator.perf_counter", lambda: next(clock))

    result = evaluator.evaluate_video(
        "fall/fall-01-cam0.mp4",
        label="fall",
        should_alarm=True,
        action_start_seconds=1.2,
        max_alarm_delay_seconds=2.0,
        activity_group="fall",
    )

    assert result.frame_count == 30
    assert result.fps == 10.0
    assert result.detected_person_frame_count == 30
    assert result.processing_fps == pytest.approx(30.0)
    assert len(result.alarm_times) == 1
    assert result.alarm_times[0] / 0.1 == pytest.approx(round(result.alarm_times[0] / 0.1))
    assert 1.2 <= result.alarm_times[0] <= 2.4
    assert result.features
    assert result.read_error is None
    assert evaluator._device == "cuda"
    assert model.devices == ["cuda"] * 30
    assert model.verbose_values == [False] * 30
    assert pose.images and capture.released

    evaluator.close()
    assert model.closed
    assert pose.closed


def test_evaluate_video_crops_urfd_rgb_before_model_inference_and_upscales():
    model, pose, _ = make_fall_components(frame_count=1)
    model.boxes_by_frame = [[(0.0, 0.0, 320.0, 240.0, 0)]]
    frame = np.zeros((240, 640, 3), dtype=np.uint8)
    frame[:, :320] = (255, 0, 0)
    frame[:, 320:] = (0, 0, 255)
    capture = FakeCapture([frame], fps=30.0)
    evaluator = make_evaluator(model, pose, capture, split_rgb=True, upscale=1.0)

    result = evaluator.evaluate_video(
        "fall/fall-01-cam0.mp4",
        label="fall",
        should_alarm=True,
        action_start_seconds=None,
        max_alarm_delay_seconds=2.0,
        activity_group="fall",
    )

    expected_right_half = frame[:, 320:]
    assert model.frames[0].shape == (240, 320, 3)
    assert np.array_equal(model.frames[0], expected_right_half)
    assert np.all(pose.images[0] == (255, 0, 0))
    assert result.read_error is None


def test_evaluate_video_upscales_the_cropped_frame():
    model, pose, _ = make_fall_components(frame_count=1)
    capture = FakeCapture([np.full((240, 640, 3), 7, dtype=np.uint8)], fps=30.0)
    evaluator = make_evaluator(model, pose, capture, split_rgb=True, upscale=2.0)

    result = evaluator.evaluate_video(
        "fall/fall-01-cam0.mp4",
        label="fall",
        should_alarm=True,
        action_start_seconds=None,
        max_alarm_delay_seconds=2.0,
        activity_group="fall",
    )

    assert model.frames[0].shape == (480, 640, 3)
    assert result.read_error is None


@pytest.mark.parametrize("failure", ["model", "pose"])
def test_evaluate_video_propagates_processing_failures_without_read_error(failure):
    model = FailingModel() if failure == "model" else FakeModel([[(10, 10, 60, 100, 0)]])
    pose = FailingPose() if failure == "pose" else FakePose([None])
    capture = FakeCapture([np.zeros((240, 320, 3), dtype=np.uint8)])
    evaluator = make_evaluator(model, pose, capture, split_rgb=False)

    with pytest.raises(RuntimeError, match=f"{failure} .* failed"):
        evaluator.evaluate_video(
            "fall/fall-01-cam0.mp4",
            label="fall",
            should_alarm=True,
            action_start_seconds=None,
            max_alarm_delay_seconds=2.0,
            activity_group="fall",
        )

    assert capture.released


def test_capture_release_failure_does_not_replace_a_prepared_result():
    model, pose, _ = make_fall_components(frame_count=1)
    capture = ReleaseFailureCapture([np.zeros((240, 320, 3), dtype=np.uint8)])
    evaluator = make_evaluator(model, pose, capture, split_rgb=False)

    result = evaluator.evaluate_video(
        "adl/adl-01-cam0.mp4",
        label="adl",
        should_alarm=False,
        action_start_seconds=None,
        max_alarm_delay_seconds=2.0,
        activity_group="basic_adl",
    )

    assert result.read_error is None
    assert capture.release_failed


def test_capture_release_failure_does_not_replace_a_processing_exception():
    capture = ReleaseFailureCapture([np.zeros((240, 320, 3), dtype=np.uint8)])
    evaluator = make_evaluator(
        FailingModel(),
        FakePose([None]),
        capture,
        split_rgb=False,
    )

    with pytest.raises(RuntimeError, match="model inference failed"):
        evaluator.evaluate_video(
            "fall/fall-01-cam0.mp4",
            label="fall",
            should_alarm=True,
            action_start_seconds=None,
            max_alarm_delay_seconds=2.0,
            activity_group="fall",
        )

    assert capture.release_failed


def test_close_attempts_model_and_pose_cleanup_independently():
    model = CloseFailureResource()
    pose = CloseFailureResource()
    evaluator = FallVideoEvaluator(
        model_path="fake-model.pt",
        _model=model,
        _pose=pose,
        _capture_factory=lambda path: FakeCapture([]),
        _torch_cuda_available=False,
    )

    evaluator.close()

    assert model.close_attempted
    assert pose.close_attempted


def test_evaluate_video_counts_a_multi_person_frame_once_and_filters_boxes():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    model = FakeModel(
        [[
            (10.0, 10.0, 40.0, 50.0, 0),
            (-20.0, -20.0, 20.0, 20.0, 0),
            (100.0, 100.0, 105.0, 105.0, 0),
            (150.0, 150.0, 250.0, 230.0, 1),
        ]]
    )
    pose = FakePose([None, None])
    capture = FakeCapture([frame], fps=30.0)
    evaluator = make_evaluator(model, pose, capture, split_rgb=False)

    result = evaluator.evaluate_video(
        "adl/adl-01-cam0.mp4",
        label="adl",
        should_alarm=False,
        action_start_seconds=None,
        max_alarm_delay_seconds=2.0,
        activity_group="basic_adl",
    )

    assert result.detected_person_frame_count == 1
    assert len(pose.images) == 2


def test_evaluate_video_records_features_only_for_alarm_events(monkeypatch):
    from deidao import fall_video_evaluator

    class FakeEngine:
        decisions = [
            FallDecision("ACTIVE", "active", features=("normal_feature",)),
            FallDecision("CONFIRMED_FALL", "falling", alarm_active=True),
            FallDecision(
                "CONFIRMED_FALL",
                "falling",
                alarm_event=True,
                alarm_active=True,
                features=("alarm_feature",),
            ),
        ]

        def __init__(self):
            self._index = 0

        def process(self, observation):
            decision = FakeEngine.decisions[self._index]
            self._index += 1
            return decision

    monkeypatch.setattr(fall_video_evaluator, "FallDetectionEngine", FakeEngine)
    model = FakeModel([[(10.0, 10.0, 60.0, 100.0, 0)]])
    pose = FakePose([None])
    frames = [np.zeros((240, 320, 3), dtype=np.uint8) for _ in range(3)]
    capture = FakeCapture(frames, fps=30.0)
    evaluator = make_evaluator(model, pose, capture, split_rgb=False)

    result = evaluator.evaluate_video(
        "fall/fall-01-cam0.mp4",
        label="fall",
        should_alarm=True,
        action_start_seconds=None,
        max_alarm_delay_seconds=2.0,
        activity_group="fall",
    )

    assert result.alarm_times == (2.0 / 30.0,)
    assert result.features == ("alarm_feature",)


def test_evaluate_video_returns_read_error_when_no_frames_can_be_read():
    model, pose, _ = make_fall_components(frame_count=1)
    capture = FakeCapture([], fps=30.0)
    evaluator = make_evaluator(model, pose, capture, split_rgb=False)

    result = evaluator.evaluate_video(
        "fall/empty.mp4",
        label="fall",
        should_alarm=True,
        action_start_seconds=None,
        max_alarm_delay_seconds=2.0,
        activity_group="fall",
    )

    assert result.read_error == "no readable video frames"
    assert result.frame_count == 0
    assert capture.released


@pytest.mark.parametrize("fps", [0.0, float("nan"), float("inf")])
def test_evaluate_video_falls_back_to_30_fps_for_invalid_metadata(fps):
    model, pose, _ = make_fall_components(frame_count=1)
    capture = FakeCapture([np.zeros((240, 320, 3), dtype=np.uint8)], fps=fps)
    evaluator = make_evaluator(model, pose, capture, split_rgb=False)

    result = evaluator.evaluate_video(
        "adl/adl-01-cam0.mp4",
        label="adl",
        should_alarm=False,
        action_start_seconds=None,
        max_alarm_delay_seconds=2.0,
        activity_group="basic_adl",
    )

    assert result.fps == 30.0


def test_evaluate_video_returns_read_error_and_releases_capture():
    error = RuntimeError("camera backend failed")
    capture = BrokenCapture(error)
    model, pose, _ = make_fall_components(frame_count=1)
    evaluator = make_evaluator(model, pose, capture, split_rgb=False)

    result = evaluator.evaluate_video(
        "fall/broken.mp4",
        label="fall",
        should_alarm=True,
        action_start_seconds=None,
        max_alarm_delay_seconds=2.0,
        activity_group="fall",
    )

    assert result.read_error == "video capture open failed: camera backend failed"
    assert result.frame_count == 0
    assert result.fps == 30.0
    assert result.alarm_times == ()
    assert result.features == ()
    assert result.processing_fps == 0.0
    assert capture.released


def test_evaluator_reuses_model_pose_and_one_capture_per_video():
    model = FakeModel([[(10.0, 10.0, 60.0, 100.0, 0)]])
    pose = FakePose([None])
    first = FakeCapture([np.zeros((240, 320, 3), dtype=np.uint8)])
    second = FakeCapture([np.zeros((240, 320, 3), dtype=np.uint8)])
    captures = [first, second]
    evaluator = FallVideoEvaluator(
        model_path="fake-model.pt",
        _model=model,
        _pose=pose,
        _capture_factory=lambda video_path: captures.pop(0),
        _torch_cuda_available=False,
    )

    for path in ("fall/one.mp4", "fall/two.mp4"):
        result = evaluator.evaluate_video(
            path,
            label="fall",
            should_alarm=True,
            action_start_seconds=None,
            max_alarm_delay_seconds=2.0,
            activity_group="fall",
        )
        assert result.read_error is None

    assert len(model.frames) == 2
    assert len(pose.images) == 2
    assert first.released
    assert second.released
    assert evaluator._device == "cpu"


def test_evaluator_rejects_non_positive_upscale():
    with pytest.raises(ValueError, match="upscale must be greater than zero"):
        FallVideoEvaluator(upscale=0.0, _model=FakeModel([[ ]]), _pose=FakePose([None]))


def test_cli_writes_json_and_markdown_and_continues_after_read_errors(tmp_path):
    cli = load_cli_module()
    dataset = tmp_path / "urfd"
    fall_dir = dataset / "fall"
    adl_dir = dataset / "adl"
    fall_dir.mkdir(parents=True)
    adl_dir.mkdir()
    (fall_dir / "fall-01-cam0.mp4").write_bytes(b"fake")
    (adl_dir / "adl-01-cam0.mp4").write_bytes(b"fake")
    manifest = {
        "dataset": "URFD",
        "camera": "cam0",
        "fps": 30.0,
        "entries": [
            {
                "path": "fall/fall-01-cam0.mp4",
                "label": "fall",
                "should_alarm": True,
                "action_start_seconds": 1.0,
                "max_alarm_delay_seconds": 2.0,
                "activity_group": "fall",
            },
            {
                "path": "adl/adl-01-cam0.mp4",
                "label": "adl",
                "should_alarm": False,
                "activity_group": "basic_adl",
            },
        ],
    }
    (dataset / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    class FakeEvaluator:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.closed = False

        def evaluate_video(self, path, **kwargs):
            read_error = None
            alarm_times = ()
            if Path(path).name.startswith("fall"):
                alarm_times = (1.2,)
            else:
                read_error = "decoder failed"
            return VideoEvaluationResult(
                path=str(path),
                alarm_times=alarm_times,
                frame_count=10,
                fps=30.0,
                detected_person_frame_count=9,
                processing_fps=20.0,
                features=("rapid_hip_descent",),
                read_error=read_error,
                **kwargs,
            )

        def close(self):
            self.closed = True

    output = tmp_path / "reports"
    exit_code = cli.main(
        [
            "--dataset",
            str(dataset),
            "--output",
            str(output),
            "--model",
            "fake.pt",
            "--split-rgb",
            "--upscale",
            "2.0",
        ],
        evaluator_factory=FakeEvaluator,
    )

    assert exit_code == 0
    payload = json.loads((output / "fall_evaluation.json").read_text(encoding="utf-8"))
    assert payload["dataset"] == "URFD"
    assert payload["camera"] == "cam0"
    assert payload.get("selected_count") == 2
    assert payload["evaluated_count"] == 1
    assert payload["overall_metrics"]["tp"] == 1
    assert payload["read_errors"] == [f"{dataset / 'adl' / 'adl-01-cam0.mp4'}: decoder failed"]
    assert [result["path"] for result in payload["results"]] == [
        str(dataset / "fall" / "fall-01-cam0.mp4"),
        str(dataset / "adl" / "adl-01-cam0.mp4"),
    ]
    assert payload["results"][0]["alarm_times"] == [1.2]
    assert (output / "fall_evaluation.md").read_text(encoding="utf-8").startswith("# URFD")


def test_cli_parse_args_exposes_all_evaluation_options():
    cli = load_cli_module()

    args = cli.parse_args(
        [
            "--dataset",
            "dataset",
            "--manifest",
            "custom.json",
            "--output",
            "reports",
            "--model",
            "model.pt",
            "--device",
            "cpu",
            "--split-rgb",
            "--upscale",
            "2.5",
            "--limit",
            "3",
            "--only-label",
            "fall",
        ]
    )

    assert args.dataset == "dataset"
    assert args.manifest == "custom.json"
    assert args.output == "reports"
    assert args.model == "model.pt"
    assert args.device == "cpu"
    assert args.split_rgb is True
    assert args.upscale == 2.5
    assert args.limit == 3
    assert args.only_label == "fall"


def write_cli_dataset(tmp_path, entries):
    dataset = tmp_path / "urfd"
    for entry in entries:
        video_path = dataset / entry["path"]
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.write_bytes(b"fake")
    manifest = {
        "dataset": "URFD",
        "camera": "cam0",
        "fps": 30.0,
        "entries": entries,
    }
    (dataset / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return dataset


def make_cli_result(path, **overrides):
    values = {
        "path": str(path),
        "label": "fall",
        "should_alarm": True,
        "alarm_times": (),
        "action_start_seconds": None,
        "max_alarm_delay_seconds": 2.0,
        "activity_group": "fall",
        "frame_count": 1,
        "fps": 30.0,
        "detected_person_frame_count": 1,
        "processing_fps": 10.0,
        "features": (),
        "read_error": None,
    }
    values.update(overrides)
    return VideoEvaluationResult(**values)


def standard_cli_entry(name, label="fall"):
    activity_group = label if label == "fall" else "basic_adl"
    return {
        "path": f"{label}/{name}.mp4",
        "label": label,
        "should_alarm": label == "fall",
        "action_start_seconds": None,
        "max_alarm_delay_seconds": 2.0,
        "activity_group": activity_group,
    }


def test_cli_applies_only_label_and_limit_before_evaluation(tmp_path):
    cli = load_cli_module()
    dataset = write_cli_dataset(
        tmp_path,
        [
            standard_cli_entry("fall-01"),
            standard_cli_entry("fall-02"),
            standard_cli_entry("adl-01", label="adl"),
        ],
    )
    evaluated_paths = []

    class RecordingEvaluator:
        def __init__(self, **kwargs):
            self.closed = False

        def evaluate_video(self, path, **kwargs):
            evaluated_paths.append(str(path))
            return make_cli_result(path, **kwargs)

        def close(self):
            self.closed = True

    exit_code = cli.main(
        [
            "--dataset",
            str(dataset),
            "--output",
            str(tmp_path / "reports"),
            "--only-label",
            "fall",
            "--limit",
            "1",
        ],
        evaluator_factory=RecordingEvaluator,
    )

    assert exit_code == 0
    assert evaluated_paths == [str(dataset / "fall" / "fall-01.mp4")]


def test_cli_returns_nonzero_for_missing_manifest_without_loading_model(tmp_path):
    cli = load_cli_module()

    def failing_factory(**kwargs):
        raise AssertionError("model must not load when the manifest is missing")

    exit_code = cli.main(
        ["--dataset", str(tmp_path / "missing"), "--output", str(tmp_path / "reports")],
        evaluator_factory=failing_factory,
    )

    assert exit_code != 0


def test_cli_returns_nonzero_when_no_entries_are_selected(tmp_path):
    cli = load_cli_module()
    dataset = write_cli_dataset(tmp_path, [standard_cli_entry("fall-01")])

    def failing_factory(**kwargs):
        raise AssertionError("model must not load when no entries are selected")

    exit_code = cli.main(
        [
            "--dataset",
            str(dataset),
            "--only-label",
            "adl",
            "--output",
            str(tmp_path / "reports"),
        ],
        evaluator_factory=failing_factory,
    )

    assert exit_code != 0


def test_cli_returns_nonzero_when_every_video_fails_to_read(tmp_path):
    cli = load_cli_module()
    dataset = write_cli_dataset(tmp_path, [standard_cli_entry("fall-01")])

    class FailingReadEvaluator:
        def __init__(self, **kwargs):
            self.closed = False

        def evaluate_video(self, path, **kwargs):
            return make_cli_result(path, read_error="decoder failed", **kwargs)

        def close(self):
            self.closed = True

    output = tmp_path / "reports"
    exit_code = cli.main(
        ["--dataset", str(dataset), "--output", str(output)],
        evaluator_factory=FailingReadEvaluator,
    )

    assert exit_code != 0
    assert (output / "fall_evaluation.json").is_file()
    assert (output / "fall_evaluation.md").is_file()


def test_cli_returns_nonzero_for_a_processing_exception_and_closes_evaluator(tmp_path, capsys):
    cli = load_cli_module()
    dataset = write_cli_dataset(tmp_path, [standard_cli_entry("fall-01")])

    class ProcessingFailureEvaluator:
        def __init__(self, **kwargs):
            self.closed = False

        def evaluate_video(self, path, **kwargs):
            raise RuntimeError("pose pipeline failed")

        def close(self):
            self.closed = True

    evaluator = ProcessingFailureEvaluator()

    def factory(**kwargs):
        return evaluator

    exit_code = cli.main(
        ["--dataset", str(dataset), "--output", str(tmp_path / "reports")],
        evaluator_factory=factory,
    )

    assert exit_code != 0
    assert "Evaluation failed: pose pipeline failed" in capsys.readouterr().err
    assert evaluator.closed


def test_cli_continues_after_per_video_frame_read_failure(tmp_path):
    cli = load_cli_module()
    entries = [standard_cli_entry("broken"), standard_cli_entry("good")]
    dataset = write_cli_dataset(tmp_path, entries)
    model = FakeModel([[(10, 10, 60, 100, 0)]])
    pose = FakePose([None])
    captures = [
        FailingReadCapture([], fps=30.0),
        FakeCapture([np.zeros((240, 320, 3), dtype=np.uint8)], fps=30.0),
    ]
    evaluator = FallVideoEvaluator(
        model_path="fake-model.pt",
        _model=model,
        _pose=pose,
        _capture_factory=lambda video_path: captures.pop(0),
        _torch_cuda_available=False,
    )

    def factory(**kwargs):
        return evaluator

    output = tmp_path / "reports"
    exit_code = cli.main(
        ["--dataset", str(dataset), "--output", str(output)],
        evaluator_factory=factory,
    )

    assert exit_code == 0
    payload = json.loads((output / "fall_evaluation.json").read_text(encoding="utf-8"))
    assert payload["read_errors"] == [
        f"{dataset / 'fall' / 'broken.mp4'}: video frame read failed: decoder failed"
    ]
    assert len(model.frames) == 1
    assert len(pose.images) == 1
