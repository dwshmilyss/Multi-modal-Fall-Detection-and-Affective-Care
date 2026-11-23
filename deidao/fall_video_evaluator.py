"""Qt-free video evaluation for the temporal fall detector."""

import math
from time import perf_counter
from typing import Callable, Optional, Sequence

import cv2
import numpy as np

from deidao.fall_engine import FallDetectionEngine, SimplePersonTracker
from deidao.fall_metrics import VideoEvaluationResult
from deidao.pose_observation import build_pose_observation


CaptureFactory = Callable[[str], object]


def crop_urfd_rgb(frame: np.ndarray) -> np.ndarray:
    """Return the exact right half of a side-by-side URFD frame."""

    if frame.ndim != 3:
        raise ValueError("frame must have shape height, width, channels")
    width = frame.shape[1]
    if width < 2:
        raise ValueError("frame width must be at least two pixels")
    return frame[:, width // 2 :]


def evaluate_alarm_times(
    alarm_times: Sequence[float],
    action_start_seconds: Optional[float],
    max_alarm_delay_seconds: float,
) -> dict:
    """Classify alarms against the inclusive timely-alarm window."""

    if action_start_seconds is None:
        return {
            "status": "undefined",
            "matched_alarm_time": None,
            "latency_seconds": None,
        }

    start = float(action_start_seconds)
    deadline = start + float(max_alarm_delay_seconds)
    alarms = tuple(float(alarm_time) for alarm_time in alarm_times)

    timely = [
        (alarm_time, alarm_time - start)
        for alarm_time in alarms
        if start <= alarm_time <= deadline
    ]
    if timely:
        matched_alarm_time, latency = min(timely, key=lambda item: item[0])
        return {
            "status": "timely",
            "matched_alarm_time": matched_alarm_time,
            "latency_seconds": latency,
        }

    late = [(alarm_time, alarm_time - start) for alarm_time in alarms if alarm_time > deadline]
    if late:
        matched_alarm_time, latency = min(late, key=lambda item: item[0])
        return {
            "status": "late",
            "matched_alarm_time": matched_alarm_time,
            "latency_seconds": latency,
        }

    return {
        "status": "missed",
        "matched_alarm_time": None,
        "latency_seconds": None,
    }


class FallVideoEvaluator:
    """Evaluate manifest videos with one shared YOLO and MediaPipe context."""

    def __init__(
        self,
        model_path: str = "model/yolov8n.pt",
        device: str = "auto",
        split_rgb: bool = False,
        upscale: float = 1.0,
        *,
        _model=None,
        _pose=None,
        _capture_factory: Optional[CaptureFactory] = None,
        _torch_cuda_available: Optional[bool] = None,
    ):
        if upscale <= 0.0 or not math.isfinite(upscale):
            raise ValueError("upscale must be greater than zero")

        self.model_path = model_path
        self.device = device
        self.split_rgb = split_rgb
        self.upscale = float(upscale)
        self._device = self._resolve_device(device, _torch_cuda_available)
        self._capture_factory = _capture_factory or self._open_capture
        self._pose_indices = None

        try:
            self._model = _model if _model is not None else self._load_yolo()
            self._pose = _pose if _pose is not None else self._load_pose()
            if _pose is not None:
                self._pose_indices = self._extract_pose_indices(_pose.PoseLandmark)
        except Exception:
            if _model is None and getattr(self, "_model", None) is not None:
                self._close_if_present(self._model)
            raise

    @staticmethod
    def _open_capture(video_path: str):
        return cv2.VideoCapture(video_path)

    @staticmethod
    def _close_if_present(resource) -> None:
        close = getattr(resource, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass

    @staticmethod
    def _resolve_device(device: str, cuda_available: Optional[bool]) -> str:
        if device != "auto":
            return device
        if cuda_available is not None:
            return "cuda" if cuda_available else "cpu"
        try:
            import torch
        except ImportError:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _load_yolo(self):
        from ultralytics import YOLO

        return YOLO(self.model_path)

    def _load_pose(self):
        import mediapipe as mp

        pose_module = mp.solutions.pose
        self._pose_indices = self._extract_pose_indices(pose_module.PoseLandmark)
        return pose_module.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1,
        )

    def close(self) -> None:
        """Release the shared model and pose context."""

        model = getattr(self, "_model", None)
        pose = getattr(self, "_pose", None)
        self._close_if_present(model)
        self._close_if_present(pose)
        self._model = None
        self._pose = None

    def evaluate_video(
        self,
        video_path: str,
        *,
        label: str,
        should_alarm: bool,
        action_start_seconds: Optional[float] = None,
        max_alarm_delay_seconds: float = 2.0,
        activity_group: str,
    ) -> VideoEvaluationResult:
        """Evaluate one file exactly once using video-relative timestamps."""

        capture = None
        frame_index = 0
        detected_person_frame_count = 0
        alarm_times = []
        feature_names = []
        feature_set = set()
        tracker = SimplePersonTracker()
        engines = {}
        fps = 30.0
        started_at = perf_counter()

        try:
            try:
                capture = self._capture_factory(str(video_path))
                opened = capture.isOpened()
            except Exception as error:
                return self._error_result(
                    video_path,
                    label,
                    should_alarm,
                    action_start_seconds,
                    max_alarm_delay_seconds,
                    activity_group,
                    frame_index,
                    fps,
                    started_at,
                    f"video capture open failed: {error}",
                )

            if not opened:
                return self._error_result(
                    video_path,
                    label,
                    should_alarm,
                    action_start_seconds,
                    max_alarm_delay_seconds,
                    activity_group,
                    frame_index,
                    fps,
                    started_at,
                    "video capture did not open",
                )

            try:
                reported_fps = float(capture.get(cv2.CAP_PROP_FPS))
                if math.isfinite(reported_fps) and reported_fps > 0.0:
                    fps = reported_fps
            except Exception as error:
                return self._error_result(
                    video_path,
                    label,
                    should_alarm,
                    action_start_seconds,
                    max_alarm_delay_seconds,
                    activity_group,
                    frame_index,
                    fps,
                    started_at,
                    f"video metadata read failed: {error}",
                )

            while True:
                try:
                    read_ok, frame = capture.read()
                except Exception as error:
                    return self._error_result(
                        video_path,
                        label,
                        should_alarm,
                        action_start_seconds,
                        max_alarm_delay_seconds,
                        activity_group,
                        frame_index,
                        fps,
                        started_at,
                        f"video frame read failed: {error}",
                    )
                if not read_ok:
                    break
                if frame is None:
                    return self._error_result(
                        video_path,
                        label,
                        should_alarm,
                        action_start_seconds,
                        max_alarm_delay_seconds,
                        activity_group,
                        frame_index,
                        fps,
                        started_at,
                        "video frame read failed: decoder returned an empty frame",
                    )

                timestamp = frame_index / fps
                prepared_frame = self._prepare_frame(frame)
                person_boxes = self._detect_people(prepared_frame)
                track_ids = tracker.assign(person_boxes, timestamp)

                if person_boxes:
                    detected_person_frame_count += 1

                for box, track_id in zip(person_boxes, track_ids):
                    engine = engines.setdefault(track_id, FallDetectionEngine())
                    decision = self._process_person(
                        engine,
                        prepared_frame,
                        box,
                        timestamp,
                    )
                    if decision.alarm_event:
                        alarm_times.append(timestamp)
                        for feature in decision.features:
                            if feature not in feature_set:
                                feature_set.add(feature)
                                feature_names.append(feature)

                active_ids = tracker.active_ids()
                for track_id, engine in engines.items():
                    if track_id not in active_ids:
                        engine.process({"timestamp": timestamp, "visibility": 0.0})

                frame_index += 1

            if frame_index == 0:
                return self._error_result(
                    video_path,
                    label,
                    should_alarm,
                    action_start_seconds,
                    max_alarm_delay_seconds,
                    activity_group,
                    frame_index,
                    fps,
                    started_at,
                    "no readable video frames",
                )

            processing_fps = self._processing_fps(frame_index, started_at)
            return VideoEvaluationResult(
                path=str(video_path),
                label=label,
                should_alarm=should_alarm,
                alarm_times=tuple(alarm_times),
                action_start_seconds=action_start_seconds,
                max_alarm_delay_seconds=max_alarm_delay_seconds,
                activity_group=activity_group,
                frame_count=frame_index,
                fps=fps,
                detected_person_frame_count=detected_person_frame_count,
                processing_fps=processing_fps,
                features=tuple(feature_names),
                read_error=None,
            )
        finally:
            if capture is not None:
                try:
                    capture.release()
                except Exception:
                    pass
            engines.clear()

    def _prepare_frame(self, frame: np.ndarray) -> np.ndarray:
        prepared = crop_urfd_rgb(frame) if self.split_rgb else frame
        if self.upscale != 1.0:
            prepared = cv2.resize(
                prepared,
                None,
                fx=self.upscale,
                fy=self.upscale,
                interpolation=cv2.INTER_LINEAR,
            )
        return prepared

    def _detect_people(self, frame: np.ndarray):
        results = self._model(frame, device=self._device, verbose=False)
        height, width = frame.shape[:2]
        person_boxes = []

        for result in results:
            boxes = result.boxes
            for bbox, object_class in zip(boxes.xyxy, boxes.cls):
                if int(object_class) != 0:
                    continue
                x1, y1, x2, y2 = map(int, bbox)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                if x2 - x1 >= 10 and y2 - y1 >= 10:
                    person_boxes.append((x1, y1, x2, y2))

        return person_boxes

    def _process_person(self, engine: FallDetectionEngine, frame: np.ndarray, box, timestamp: float):
        x1, y1, x2, y2 = box
        person_image = frame[y1:y2, x1:x2].copy()
        pose_result = self._pose.process(cv2.cvtColor(person_image, cv2.COLOR_BGR2RGB))

        if not pose_result.pose_landmarks:
            return engine.process({"timestamp": timestamp, "visibility": 0.0})

        indices = self._pose_indices
        if indices is None:
            raise RuntimeError("MediaPipe pose landmark indices are unavailable")
        observation = build_pose_observation(
            landmarks=pose_result.pose_landmarks.landmark,
            nose_index=indices["nose"],
            left_shoulder_index=indices["left_shoulder"],
            right_shoulder_index=indices["right_shoulder"],
            left_hip_index=indices["left_hip"],
            right_hip_index=indices["right_hip"],
            box=box,
            timestamp=timestamp,
        )
        if observation is None:
            return engine.process({"timestamp": timestamp, "visibility": 0.0})
        return engine.process(observation)

    @staticmethod
    def _extract_pose_indices(landmarks) -> dict:
        def index(landmark):
            return int(getattr(landmark, "value", landmark))

        return {
            "nose": index(landmarks.NOSE),
            "left_shoulder": index(landmarks.LEFT_SHOULDER),
            "right_shoulder": index(landmarks.RIGHT_SHOULDER),
            "left_hip": index(landmarks.LEFT_HIP),
            "right_hip": index(landmarks.RIGHT_HIP),
        }

    @staticmethod
    def _processing_fps(frame_count: int, started_at: float) -> float:
        elapsed = max(0.0, perf_counter() - started_at)
        return frame_count / elapsed if frame_count and elapsed > 0.0 else 0.0

    @staticmethod
    def _error_result(
        video_path,
        label,
        should_alarm,
        action_start_seconds,
        max_alarm_delay_seconds,
        activity_group,
        frame_count,
        fps,
        started_at,
        message,
    ) -> VideoEvaluationResult:
        return VideoEvaluationResult(
            path=str(video_path),
            label=label,
            should_alarm=should_alarm,
            alarm_times=(),
            action_start_seconds=action_start_seconds,
            max_alarm_delay_seconds=max_alarm_delay_seconds,
            activity_group=activity_group,
            frame_count=frame_count,
            fps=fps,
            detected_person_frame_count=0,
            processing_fps=FallVideoEvaluator._processing_fps(frame_count, started_at),
            features=(),
            read_error=message,
        )
