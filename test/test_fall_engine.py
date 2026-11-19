import math

import pytest

from deidao.fall_engine import (
    FallDetectionEngine,
    FallEngineSettings,
    SimplePersonTracker,
)


def observation(timestamp, *, angle=5.0, hip_y=220.0, nose_y=80.0, box=None, visibility=0.95):
    if box is None:
        box = (100.0, 50.0, 180.0, 450.0)
    return {
        "timestamp": timestamp,
        "torso_angle": angle,
        "hip_y": hip_y,
        "nose_y": nose_y,
        "bbox": box,
        "visibility": visibility,
    }


def feed_upright(engine, start, seconds=1.2, step=1.0 / 30.0):
    timestamp = start
    while timestamp < start + seconds:
        engine.process(observation(timestamp))
        timestamp += step
    return timestamp


def feed_fast_fall(engine, start, fps):
    step = 1.0 / fps
    duration = 0.18
    frames = max(2, int(round(duration * fps)) + 1)
    events = []
    for index in range(1, frames + 1):
        progress = index / (frames - 1)
        result = engine.process(
            observation(
                start + index * step,
                angle=5 + progress * 75,
                hip_y=220 + progress * 45,
                nose_y=80 + progress * 90,
                box=(100, 100, 100 + 240 * progress, 285),
            )
        )
        if result.alarm_event:
            events.append(result)
    return events


def test_slow_intentional_lying_is_recognized_without_alarm():
    engine = FallDetectionEngine()
    now = feed_upright(engine, 0.0)
    duration = 1.8
    frames = 54

    for index in range(1, frames + 1):
        progress = index / frames
        now += duration / frames
        result = engine.process(
            observation(
                now,
                angle=5 + progress * 75,
                hip_y=220 + progress * 45,
                nose_y=80 + progress * 90,
                box=(100, 100, 100 + 240 * progress, 285),
            )
        )
        assert not result.alarm_event
        assert not result.alarm_active

    for index in range(120):
        now += 1.0 / 30.0
        result = engine.process(
            observation(
                now,
                angle=80,
                hip_y=265,
                nose_y=170,
                box=(100, 100, 340, 285),
            )
        )
        assert not result.alarm_event
        assert not result.alarm_active

    assert result.state == "LYING"
    assert result.label == "平躺 (Lying)"


def test_fast_fall_is_confirmed_once_and_holds_alarm():
    engine = FallDetectionEngine()
    now = feed_upright(engine, 0.0)
    events = feed_fast_fall(engine, now, fps=30)

    assert len(events) == 1
    event = events[0]
    assert event.state == "CONFIRMED_FALL"
    assert event.label == "跌倒中 (Falling!)"
    assert event.confidence >= engine.settings.alert_score

    hold_result = None
    for index in range(1, 120):
        hold_result = engine.process(
            observation(
                now + 0.18 + index / 30.0,
                angle=80,
                hip_y=265,
                nose_y=170,
                box=(100, 100, 340, 285),
            )
        )
        if index < 5:
            assert hold_result.alarm_active
        assert not hold_result.alarm_event

    assert hold_result.state == "LYING"
    assert not hold_result.alarm_active


def test_fast_sitting_reaches_stable_sitting_without_alarm():
    engine = FallDetectionEngine()
    now = feed_upright(engine, 0.0)
    duration = 0.45
    frames = 18

    for index in range(1, frames + 1):
        progress = index / frames
        result = engine.process(
            observation(
                now + duration * index / frames,
                angle=5 + progress * 32,
                hip_y=220 + progress * 110,
                nose_y=80 + progress * 40,
                box=(90, 90, 90 + 110 * progress, 450),
            )
        )
        assert not result.alarm_event

    for index in range(30):
        result = engine.process(
            observation(
                now + 0.45 + index / 30.0,
                angle=37,
                hip_y=330,
                nose_y=120,
                box=(90, 90, 200, 450),
            )
        )

    assert result.state == "SITTING"
    assert result.label == "坐姿 (Sitting)"
    assert not result.alarm_event
    assert not result.alarm_active


def test_low_hip_without_dynamic_evidence_does_not_bypass_consensus():
    engine = FallDetectionEngine()
    feed_upright(engine, 0.0)

    first_after_gap = engine.process(
        observation(
            0.8,
            angle=78,
            hip_y=260,
            nose_y=165,
            box=(100, 100, 340, 285),
        )
    )
    assert not first_after_gap.alarm_event

    result = first_after_gap
    for index in range(20):
        result = engine.process(
            observation(
                0.8 + index / 30.0,
                angle=78,
                hip_y=260,
                nose_y=165,
                box=(100, 100, 340, 285),
            )
        )

    assert result.state == "LYING"
    assert not result.alarm_event


@pytest.mark.parametrize("fps", [15, 30])
def test_fall_decision_is_normalized_by_elapsed_time(fps):
    engine = FallDetectionEngine()
    now = feed_upright(engine, 0.0, step=1.0 / fps)
    events = feed_fast_fall(engine, now, fps=fps)
    assert len(events) == 1


def test_missing_pose_decays_a_fall_candidate():
    engine = FallDetectionEngine()
    now = feed_upright(engine, 0.0)

    candidate_result = None
    for index in range(1, 7):
        progress = index / 6
        candidate_result = engine.process(
            observation(
                now + index / 30.0,
                angle=5 + progress * 78,
                hip_y=220 + progress * 45,
                nose_y=80 + progress * 90,
                box=(100, 100, 100 + 240 * progress, 285),
            )
        )

    assert candidate_result.state in {"POTENTIAL_FALL", "CONFIRMED_FALL"}
    for index in range(10):
        result = engine.process(
            observation(
                now + 0.2 + index / 30.0,
                visibility=0.2,
            )
        )

    assert result.candidate_score < engine.settings.alert_score
    assert result.state in {"ACTIVE", "NORMAL"}


def test_simple_person_tracker_keeps_ids_across_small_moves():
    tracker = SimplePersonTracker()
    first = tracker.assign([(100, 100, 180, 220)], timestamp=0.0)
    second = tracker.assign([(105, 104, 185, 224)], timestamp=0.1)
    separated = tracker.assign([(105, 104, 185, 224), (400, 100, 480, 220)], timestamp=0.2)

    assert first == [1]
    assert second == [1]
    assert separated[0] == 1
    assert separated[1] == 2


def test_simple_person_tracker_preserves_detection_order_when_later_box_matches():
    tracker = SimplePersonTracker()
    tracker.assign([(100, 100, 180, 220), (400, 100, 480, 220)], timestamp=0.0)

    # The second old track has a much higher overlap with the second new box.
    # Greedy matching must not reorder the IDs returned for the input boxes.
    ids = tracker.assign([(120, 105, 205, 228), (405, 102, 485, 220)], timestamp=0.1)

    assert ids == [1, 2]


def test_settings_expose_time_and_body_normalized_thresholds():
    settings = FallEngineSettings()
    assert settings.angular_speed_threshold > 0
    assert 0 < settings.hip_drop_threshold < 1
    assert settings.candidate_hold_seconds > 0
    assert not math.isnan(settings.alert_score)


def test_main_detection_thread_defaults_to_temporal_implementation():
    from deidao.detector import DetectionThread, LegacyDetectionThread, TemporalDetectionThread

    assert issubclass(TemporalDetectionThread, LegacyDetectionThread)
    assert DetectionThread is TemporalDetectionThread
