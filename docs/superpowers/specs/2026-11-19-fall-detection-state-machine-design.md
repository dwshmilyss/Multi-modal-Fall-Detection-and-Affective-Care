# Fall Detection State Machine Design

## Goal

Reduce false alarms from sitting, intentional lying, bending, and camera/scale changes while retaining fast confirmation for genuine falls. The work combines the immediate rule fixes from option A with the testable temporal state machine from option B.

## Current Problems

The current detector bypasses its documented two-feature consensus when `hip_low` is true, mixes temporal state between people, uses pixel velocities that change with resolution and distance, assumes the camera FPS is accurate, and clears fall candidates as soon as a sitting heuristic matches. Its alarm flag also returns false on the next frame.

## Architecture

Add `deidao/fall_engine.py`, a Qt-free module with three small units:

- `SimplePersonTracker`: assigns stable IDs to person boxes using greedy IoU matching.
- `FallDetectionEngine`: maintains one person's timestamped pose history, upright height baseline, normalized motion features, posture state, and alarm hold state.
- `FallEngineSettings`: centralizes thresholds in real time units and body-relative scale.

`DetectionThread` remains responsible for camera I/O, YOLO person detection, MediaPipe pose, drawing, recording, and notification. It computes raw coordinates and delegates each tracked person to a `FallDetectionEngine`.

## Decision Model

The engine computes features over real elapsed time:

- torso angular speed in degrees per second;
- hip and head descent in standing-height units per second;
- bounding-box aspect-ratio change over about half a second;
- terminal hip height relative to an upright baseline;
- posture stability from low angular and vertical speed.

Each observation contributes to a weighted score. Dynamic evidence comes from rapid torso rotation and rapid hip/head descent. Terminal evidence comes from a low hip and lying-like body shape. A fall alarm requires sustained score at or above the threshold, at least one dynamic feature, and terminal low posture. This removes the single-feature `hip_low` bypass.

The engine recognizes stable sitting after chair-height posture and low motion persist briefly. It recognizes stable lying without alerting when the transition was slow. A genuine fast fall enters `CONFIRMED_FALL`, emits one alarm event, and keeps `alarm_active` true for a short hold period so the UI does not flash red for only one frame.

## Error Handling

Missing or low-visibility pose observations decay candidates rather than erasing useful history immediately. Large timestamp gaps clear stale motion history. Person tracks that disappear are marked missing and reset candidate state after a short grace period.

## Testing

Add pure unit tests for intentional slow lying, rapid fall confirmation, rapid sitting, low-hip-only suppression, FPS-independent timing, and tracker ID continuity. Run focused fall tests and the available project tests. Keep the detector integration import-light enough that algorithm regressions do not require opening a camera.

