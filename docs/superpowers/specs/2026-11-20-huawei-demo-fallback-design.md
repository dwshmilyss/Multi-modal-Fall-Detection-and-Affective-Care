# Huawei Demo Fallback Design

## Goal

When Huawei Band 9 is configured but no real device message is available, the health dashboard must retain the original simulator refresh behavior so a competition demo remains usable.

## Behavior

- `mode: huawei_band` remains valid.
- `huawei_band.fallback_to_simulator` defaults to `true`.
- On startup, the dashboard subscribes to MQTT and simultaneously starts the original 2-second simulator refresh.
- The status explains that the dashboard is connected/waiting while showing simulated demo data.
- When the first valid real MQTT `HealthData` arrives, the dashboard stops simulator updates and applies only real data.
- Subsequent real messages continue to render directly; simulator randomization never overwrites them.
- If MQTT cannot connect or no message arrives, simulator refresh continues.

## Non-Goals

- No fake device-connected status.
- No synthetic MQTT messages.
- No persistence of demo data as real health history after real data arrives.

## Verification

Add tests for:

1. Starting Huawei mode with fallback enabled starts simulator refresh even when MQTT connects.
2. The first real message stops simulator refresh and applies real data.
3. Disabling fallback preserves strict waiting behavior when MQTT connects.
