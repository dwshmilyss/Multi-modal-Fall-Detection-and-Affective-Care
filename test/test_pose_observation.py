import math

import pytest

from deidao.pose_observation import build_pose_observation


BOX = (100.0, 50.0, 240.0, 450.0)


class FakeLandmark:
    def __init__(self, x, y, visibility):
        self.x = x
        self.y = y
        self.visibility = visibility


def make_landmarks(
    nose,
    left_shoulder,
    right_shoulder,
    left_hip,
    right_hip,
):
    values = (nose, left_shoulder, right_shoulder, left_hip, right_hip)
    return [FakeLandmark(*value) for value in values]


def upright_landmarks():
    return make_landmarks(
        nose=(0.55, 0.12, 0.97),
        left_shoulder=(0.30, 0.30, 0.90),
        right_shoulder=(0.70, 0.30, 0.95),
        left_hip=(0.32, 0.65, 0.80),
        right_hip=(0.68, 0.65, 0.85),
    )


def test_valid_pose_becomes_temporal_engine_observation():
    landmarks = upright_landmarks()

    observation = build_pose_observation(
        landmarks=landmarks,
        nose_index=0,
        left_shoulder_index=1,
        right_shoulder_index=2,
        left_hip_index=3,
        right_hip_index=4,
        box=BOX,
        timestamp=12.5,
    )

    assert observation is not None
    assert observation["timestamp"] == 12.5
    assert observation["torso_angle"] == pytest.approx(90.0)
    assert observation["hip_y"] == pytest.approx(310.0)
    assert observation["nose_y"] == pytest.approx(98.0)
    assert observation["bbox"] == (100.0, 50.0, 240.0, 450.0)
    assert observation["visibility"] == pytest.approx(0.894)


@pytest.mark.parametrize(
    "box",
    [
        (100.0, 50.0, 100.0, 450.0),
        (100.0, 50.0, 90.0, 450.0),
        (100.0, 50.0, 240.0, 50.0),
        (100.0, 50.0, 240.0, 20.0),
    ],
)
def test_zero_size_or_negative_box_returns_none(box):
    observation = build_pose_observation(
        landmarks=upright_landmarks(),
        nose_index=0,
        left_shoulder_index=1,
        right_shoulder_index=2,
        left_hip_index=3,
        right_hip_index=4,
        box=box,
        timestamp=1.0,
    )

    assert observation is None


@pytest.mark.parametrize(
    "shoulder_y,hip_y",
    [
        (0.30, 0.65),
        (0.65, 0.30),
        (0.25, 0.75),
        (0.75, 0.25),
    ],
)
def test_torso_angle_stays_within_right_angle(shoulder_y, hip_y):
    landmarks = make_landmarks(
        nose=(0.50, 0.10, 1.0),
        left_shoulder=(0.25, shoulder_y, 1.0),
        right_shoulder=(0.75, shoulder_y, 1.0),
        left_hip=(0.25, hip_y, 1.0),
        right_hip=(0.75, hip_y, 1.0),
    )

    observation = build_pose_observation(
        landmarks=landmarks,
        nose_index=0,
        left_shoulder_index=1,
        right_shoulder_index=2,
        left_hip_index=3,
        right_hip_index=4,
        box=BOX,
        timestamp=0.0,
    )

    assert observation is not None
    angle = observation["torso_angle"]
    assert 0.0 <= angle <= 90.0
    assert math.isfinite(angle)


def test_torso_angle_matches_temporal_formula():
    upright = make_landmarks(
        nose=(0.50, 0.20, 1.0),
        left_shoulder=(0.25, 0.30, 1.0),
        right_shoulder=(0.75, 0.30, 1.0),
        left_hip=(0.25, 0.65, 1.0),
        right_hip=(0.75, 0.65, 1.0),
    )
    upright_observation = build_pose_observation(
        landmarks=upright,
        nose_index=0,
        left_shoulder_index=1,
        right_shoulder_index=2,
        left_hip_index=3,
        right_hip_index=4,
        box=BOX,
        timestamp=0.0,
    )

    assert upright_observation is not None
    assert upright_observation["torso_angle"] == 90.0

    slanted = make_landmarks(
        nose=(0.60, 0.15, 1.0),
        left_shoulder=(0.40, 0.50, 1.0),
        right_shoulder=(0.90, 0.70, 1.0),
        left_hip=(0.10, 0.20, 1.0),
        right_hip=(0.60, 0.40, 1.0),
    )
    slanted_observation = build_pose_observation(
        landmarks=slanted,
        nose_index=0,
        left_shoulder_index=1,
        right_shoulder_index=2,
        left_hip_index=3,
        right_hip_index=4,
        box=BOX,
        timestamp=0.0,
    )
    shoulder_center = (0.65 * 140.0, 0.60 * 400.0)
    hip_center = (0.35 * 140.0, 0.30 * 400.0)
    expected_angle = abs(90.0 - math.degrees(math.atan2(
        shoulder_center[1] - hip_center[1],
        shoulder_center[0] - hip_center[0],
    )))

    assert slanted_observation is not None
    assert slanted_observation["torso_angle"] == pytest.approx(expected_angle)


def test_arguments_are_keyword_only():
    with pytest.raises(TypeError):
        build_pose_observation(
            upright_landmarks(),
            0,
            1,
            2,
            3,
            4,
            BOX,
            1.0,
        )
