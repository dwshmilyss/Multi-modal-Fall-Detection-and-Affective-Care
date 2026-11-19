"""Qt-free temporal fall detection and lightweight person tracking.

The module intentionally avoids camera, model, and UI imports.  This makes the
fall decision testable from synthetic timestamped observations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class FallEngineSettings:
    """Thresholds expressed in real time and body-relative units."""

    angular_speed_threshold: float = 95.0
    hip_drop_threshold: float = 0.45
    head_drop_threshold: float = 0.50
    bbox_ratio_change_threshold: float = 1.5
    alert_score: float = 5.0
    candidate_hold_seconds: float = 0.12
    alarm_hold_seconds: float = 3.0
    stable_posture_seconds: float = 0.50
    torso_angle_threshold: float = 35.0
    sitting_hip_min: float = 0.24
    sitting_hip_max: float = 0.48
    lying_hip_max: float = 0.22
    lying_bbox_ratio_min: float = 0.90
    motion_window_seconds: float = 0.55
    stale_gap_seconds: float = 0.50
    minimum_visibility: float = 0.40
    missing_score_decay: float = 30.0


@dataclass(frozen=True)
class FallDecision:
    state: str
    label: str
    candidate_score: float = 0.0
    confidence: float = 0.0
    alarm_event: bool = False
    alarm_active: bool = False
    features: Tuple[str, ...] = field(default_factory=tuple)


class FallDetectionEngine:
    """Temporal fall state machine for one tracked person."""

    _LABELS = {
        "NORMAL": "站立 (Normal)",
        "ACTIVE": "弯腰/活动中 (Active)",
        "SITTING": "坐姿 (Sitting)",
        "LYING": "平躺 (Lying)",
        "POTENTIAL_FALL": "疑似跌倒 (Potential Fall)",
        "CONFIRMED_FALL": "跌倒中 (Falling!)",
    }

    def __init__(self, settings: Optional[FallEngineSettings] = None):
        self.settings = settings or FallEngineSettings()
        self._angle_history: List[Tuple[float, float]] = []
        self._hip_history: List[Tuple[float, float]] = []
        self._head_history: List[Tuple[float, float]] = []
        self._bbox_ratio_history: List[Tuple[float, float]] = []
        self._last_timestamp: Optional[float] = None
        self._upright_height_px: Optional[float] = None
        self._stable_since: Optional[float] = None
        self._candidate_since: Optional[float] = None
        self._candidate_score = 0.0
        self._state = "NORMAL"
        self._alarm_until: Optional[float] = None
        self._last_alarm_timestamp: Optional[float] = None

    def process(self, observation: Optional[dict]) -> FallDecision:
        """Process one timestamped observation and return the current decision."""
        if not observation:
            return self._process_missing(None)

        timestamp = float(observation["timestamp"])
        visibility = float(observation.get("visibility", 1.0))
        if visibility < self.settings.minimum_visibility:
            return self._process_missing(timestamp)

        if self._last_timestamp is None:
            self._last_timestamp = timestamp
        elapsed = timestamp - self._last_timestamp
        if elapsed < 0:
            elapsed = 0.0
        if elapsed > self.settings.stale_gap_seconds:
            self._clear_motion_history()
            self._candidate_since = None
            self._candidate_score = 0.0
        self._last_timestamp = timestamp

        torso_angle = max(0.0, min(90.0, float(observation["torso_angle"])))
        hip_y = float(observation["hip_y"])
        nose_y = float(observation["nose_y"])
        left, top, right, bottom = map(float, observation["bbox"])
        bbox_width = max(1.0, right - left)
        bbox_height = max(1.0, bottom - top)
        bbox_ratio = bbox_width / bbox_height

        if self._upright_height_px is None:
            self._upright_height_px = bbox_height
        elif torso_angle < 25.0 and bbox_ratio < 0.70:
            # Preserve the baseline while sitting or lying, but adapt when the
            # person changes distance from the camera while upright.
            previous = self._upright_height_px
            self._upright_height_px = previous * 0.80 + bbox_height * 0.20

        body_height = max(80.0, self._upright_height_px)

        self._append_capped(self._angle_history, timestamp, torso_angle)
        self._append_capped(self._hip_history, timestamp, hip_y)
        self._append_capped(self._head_history, timestamp, nose_y)
        self._append_capped(self._bbox_ratio_history, timestamp, bbox_ratio)

        angular_speed = self._angular_speed(timestamp)
        hip_drop_speed = self._descent_speed(self._hip_history, timestamp, body_height)
        head_drop_speed = self._descent_speed(self._head_history, timestamp, body_height)
        bbox_ratio_change = self._bbox_ratio_change(timestamp)
        hip_height_ratio = max(0.0, min(1.0, (bottom - hip_y) / body_height))

        score, dynamic_count, features = self._score_observation(
            torso_angle=torso_angle,
            angular_speed=angular_speed,
            hip_drop_speed=hip_drop_speed,
            head_drop_speed=head_drop_speed,
            bbox_ratio_change=bbox_ratio_change,
            hip_height_ratio=hip_height_ratio,
            bbox_ratio=bbox_ratio,
        )

        unstable = (
            angular_speed > 50.0
            or hip_drop_speed > 0.18
            or head_drop_speed > 0.25
        )
        if unstable or self._stable_since is None:
            self._stable_since = timestamp

        stable_duration = timestamp - self._stable_since
        sitting_target = (
            self.settings.sitting_hip_min
            <= hip_height_ratio
            <= self.settings.sitting_hip_max
            and torso_angle < 70.0
        )
        lying_target = (
            hip_height_ratio <= self.settings.lying_hip_max
            and torso_angle >= self.settings.torso_angle_threshold
        )
        stable_sitting = sitting_target and stable_duration >= self.settings.stable_posture_seconds
        stable_lying = lying_target and stable_duration >= self.settings.stable_posture_seconds

        if stable_sitting:
            self._candidate_since = None
            self._candidate_score = 0.0
            self._state = "SITTING"
        elif timestamp < (self._alarm_until or float("-inf")):
            self._state = "CONFIRMED_FALL"
        elif stable_lying:
            self._candidate_since = None
            self._candidate_score = 0.0
            self._state = "LYING"
        elif score >= self.settings.alert_score and dynamic_count > 0 and lying_target:
            if self._candidate_since is None:
                self._candidate_since = timestamp
                self._candidate_score = score
            else:
                self._candidate_score = max(self._candidate_score, score)

            held = timestamp - self._candidate_since
            if held >= self.settings.candidate_hold_seconds:
                self._state = "CONFIRMED_FALL"
                self._alarm_until = timestamp + self.settings.alarm_hold_seconds
                self._last_alarm_timestamp = timestamp
                return FallDecision(
                    state=self._state,
                    label=self._LABELS[self._state],
                    candidate_score=self._candidate_score,
                    confidence=score,
                    alarm_event=True,
                    alarm_active=True,
                    features=tuple(features),
                )
            self._state = "POTENTIAL_FALL"
        else:
            self._candidate_since = None
            self._candidate_score = 0.0
            if torso_angle < 20.0 and angular_speed < 40.0:
                self._state = "NORMAL"
            elif torso_angle >= 65.0 and angular_speed < 35.0 and hip_height_ratio <= 0.32:
                self._state = "LYING"
            else:
                self._state = "ACTIVE"

        alarm_active = timestamp < (self._alarm_until or float("-inf"))
        if alarm_active and self._state != "CONFIRMED_FALL":
            self._state = "CONFIRMED_FALL"

        return FallDecision(
            state=self._state,
            label=self._LABELS[self._state],
            candidate_score=self._candidate_score,
            confidence=score,
            alarm_event=False,
            alarm_active=alarm_active,
            features=tuple(features),
        )

    def _process_missing(self, timestamp: Optional[float]) -> FallDecision:
        if timestamp is not None:
            if self._last_timestamp is None:
                self._last_timestamp = timestamp
            elapsed = max(0.0, timestamp - self._last_timestamp)
            self._last_timestamp = timestamp
            self._candidate_score = max(
                0.0,
                self._candidate_score - elapsed * self.settings.missing_score_decay,
            )
            if self._candidate_score < self.settings.alert_score:
                self._candidate_since = None
            if elapsed > self.settings.stale_gap_seconds:
                self._clear_motion_history()
                self._candidate_score = 0.0
                self._candidate_since = None
                self._stable_since = None

        alarm_active = False
        if timestamp is not None and timestamp < (self._alarm_until or float("-inf")):
            alarm_active = True
            self._state = "CONFIRMED_FALL"
        elif self._candidate_score >= self.settings.alert_score:
            self._state = "POTENTIAL_FALL"
        elif self._state in {"POTENTIAL_FALL", "CONFIRMED_FALL"}:
            self._state = "ACTIVE"

        return FallDecision(
            state=self._state,
            label=self._LABELS[self._state],
            candidate_score=self._candidate_score,
            confidence=self._candidate_score,
            alarm_event=False,
            alarm_active=alarm_active,
        )

    def _score_observation(
        self,
        *,
        torso_angle: float,
        angular_speed: float,
        hip_drop_speed: float,
        head_drop_speed: float,
        bbox_ratio_change: float,
        hip_height_ratio: float,
        bbox_ratio: float,
    ):
        score = 0.0
        dynamic_count = 0
        features: List[str] = []

        if torso_angle >= self.settings.torso_angle_threshold and angular_speed >= self.settings.angular_speed_threshold:
            score += 3.0
            dynamic_count += 1
            features.append("rapid_torso_rotation")

        if hip_drop_speed >= self.settings.hip_drop_threshold:
            score += 3.0
            dynamic_count += 1
            features.append("rapid_hip_descent")

        if (
            head_drop_speed >= self.settings.head_drop_threshold
            and hip_drop_speed >= self.settings.hip_drop_threshold * 0.55
        ):
            score += 2.0
            dynamic_count += 1
            features.append("rapid_head_and_hip_descent")

        if (
            bbox_ratio_change >= self.settings.bbox_ratio_change_threshold
            and torso_angle >= self.settings.torso_angle_threshold
        ):
            score += 1.0
            features.append("bbox_shape_change")

        terminal_low = hip_height_ratio <= self.settings.lying_hip_max
        if terminal_low and torso_angle >= self.settings.torso_angle_threshold:
            score += 2.0
            features.append("terminal_low_hip")

        if terminal_low and bbox_ratio >= self.settings.lying_bbox_ratio_min:
            score += 1.0
            features.append("lying_shape")

        return score, dynamic_count, features

    def _angular_speed(self, timestamp: float) -> float:
        previous = self._sample_before(self._angle_history, timestamp, minimum_age=0.08)
        if previous is None:
            return 0.0
        old_timestamp, old_angle = previous
        elapsed = max(1e-6, timestamp - old_timestamp)
        return abs(self._angle_history[-1][1] - old_angle) / elapsed

    def _descent_speed(self, history: List[Tuple[float, float]], timestamp: float, body_height: float) -> float:
        previous = self._sample_before(history, timestamp, minimum_age=0.10)
        if previous is None:
            return 0.0
        old_timestamp, old_y = previous
        elapsed = max(1e-6, timestamp - old_timestamp)
        descent_px_per_sec = max(0.0, history[-1][1] - old_y) / elapsed
        return descent_px_per_sec / body_height

    def _bbox_ratio_change(self, timestamp: float) -> float:
        if not self._bbox_ratio_history:
            return 1.0
        old_timestamp, old_ratio = self._bbox_ratio_history[0]
        if old_ratio <= 0.01 or timestamp - old_timestamp < 0.08:
            return 1.0
        return self._bbox_ratio_history[-1][1] / old_ratio

    @staticmethod
    def _sample_before(
        history: Sequence[Tuple[float, float]],
        timestamp: float,
        minimum_age: float,
    ) -> Optional[Tuple[float, float]]:
        for old_timestamp, value in reversed(history[:-1]):
            age = timestamp - old_timestamp
            if age >= minimum_age:
                return old_timestamp, value
        return None

    def _append_capped(self, history: List[Tuple[float, float]], timestamp: float, value: float) -> None:
        history.append((timestamp, value))
        cutoff = timestamp - self.settings.motion_window_seconds
        while history and history[0][0] < cutoff:
            history.pop(0)

    def _clear_motion_history(self) -> None:
        self._angle_history.clear()
        self._hip_history.clear()
        self._head_history.clear()
        self._bbox_ratio_history.clear()


class SimplePersonTracker:
    """Greedy IoU tracker that is sufficient for one fixed camera view."""

    def __init__(self, iou_threshold: float = 0.15, max_missing: int = 8):
        self.iou_threshold = iou_threshold
        self.max_missing = max_missing
        self._tracks: Dict[int, Tuple[float, float, float, float]] = {}
        self._missed: Dict[int, int] = {}
        self._next_id = 1

    def assign(
        self,
        boxes: Sequence[Tuple[float, float, float, float]],
        timestamp: float,
    ) -> List[int]:
        matches: Dict[int, int] = {}
        unused_track_ids = set(self._tracks)
        unused_box_indices = set(range(len(boxes)))

        while unused_box_indices and unused_track_ids:
            best = None
            for box_index in unused_box_indices:
                box = boxes[box_index]
                for track_id in unused_track_ids:
                    overlap = self._iou(box, self._tracks[track_id])
                    if overlap < self.iou_threshold:
                        continue
                    if best is None or overlap > best[0]:
                        best = (overlap, box_index, track_id)
            if best is None:
                break
            _, box_index, track_id = best
            unused_box_indices.remove(box_index)
            unused_track_ids.remove(track_id)
            matches[box_index] = track_id
            self._tracks[track_id] = box
            self._missed[track_id] = 0

        assigned: List[int] = []
        for box_index, box in enumerate(boxes):
            if box_index in matches:
                assigned.append(matches[box_index])
                continue
            track_id = self._next_id
            self._next_id += 1
            self._tracks[track_id] = box
            self._missed[track_id] = 0
            assigned.append(track_id)

        for track_id in unused_track_ids:
            self._missed[track_id] = self._missed.get(track_id, 0) + 1
            if self._missed[track_id] > self.max_missing:
                self._tracks.pop(track_id, None)
                self._missed.pop(track_id, None)

        return assigned

    def active_ids(self) -> set:
        """Return IDs that have not exceeded the missing-frame allowance."""

        return set(self._tracks)

    @staticmethod
    def _iou(first: Sequence[float], second: Sequence[float]) -> float:
        fx1, fy1, fx2, fy2 = first
        sx1, sy1, sx2, sy2 = second
        inter_width = min(fx2, sx2) - max(fx1, sx1)
        inter_height = min(fy2, sy2) - max(fy1, sy1)
        if inter_width <= 0 or inter_height <= 0:
            return 0.0
        intersection = inter_width * inter_height
        first_area = max(0.0, fx2 - fx1) * max(0.0, fy2 - fy1)
        second_area = max(0.0, sx2 - sx1) * max(0.0, sy2 - sy1)
        union = first_area + second_area - intersection
        return intersection / union if union > 0 else 0.0
