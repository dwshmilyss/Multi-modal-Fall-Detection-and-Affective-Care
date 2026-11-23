"""Pure MediaPipe-landmark to fall-engine observation conversion."""

import math


def build_pose_observation(
    *,
    landmarks,
    nose_index,
    left_shoulder_index,
    right_shoulder_index,
    left_hip_index,
    right_hip_index,
    box,
    timestamp,
):
    """Build one temporal observation from indexed normalized landmarks.

    ``box`` is the full-frame detection box from which landmark pixels are
    derived.  Return ``None`` when the box has no positive width or height.
    """
    x1, y1, x2, y2 = box
    box_width = x2 - x1
    box_height = y2 - y1
    if box_width <= 0 or box_height <= 0:
        return None

    left_shoulder = landmarks[left_shoulder_index]
    right_shoulder = landmarks[right_shoulder_index]
    left_hip = landmarks[left_hip_index]
    right_hip = landmarks[right_hip_index]
    nose = landmarks[nose_index]

    shoulder_center = (
        (left_shoulder.x * box_width + right_shoulder.x * box_width) / 2.0,
        (left_shoulder.y * box_height + right_shoulder.y * box_height) / 2.0,
    )
    hip_center = (
        (left_hip.x * box_width + right_hip.x * box_width) / 2.0,
        (left_hip.y * box_height + right_hip.y * box_height) / 2.0,
    )
    angle = abs(90.0 - math.degrees(math.atan2(
        hip_center[1] - shoulder_center[1],
        hip_center[0] - shoulder_center[0],
    )))
    angle = max(0.0, min(90.0, angle))

    key_indices = (
        nose_index,
        left_shoulder_index,
        right_shoulder_index,
        left_hip_index,
        right_hip_index,
    )
    visibility = sum(
        landmarks[index].visibility for index in key_indices
    ) / len(key_indices)

    return {
        "timestamp": timestamp,
        "torso_angle": float(angle),
        "hip_y": float(y1 + hip_center[1]),
        "nose_y": float(y1 + nose.y * box_height),
        "bbox": (float(x1), float(y1), float(x2), float(y2)),
        "visibility": float(visibility),
    }
