# Huawei Band MQTT Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Connect Huawei Band 9 health data through Gadgetbridge's exported SQLite database and MQTT without replacing real readings with simulator values.

**Architecture:** A dedicated bridge reads the latest supported metrics from `Gadgetbridge.db` and publishes normalized JSON. `HuaweiBandAdapter` subscribes with the configured topic, converts messages to `HealthData`, and emits data from its MQTT thread. `HealthDashboardDialog` forwards callbacks through Qt signals and renders only received values.

**Tech Stack:** Python 3.11+, PyQt6 signals, `paho-mqtt`, SQLite3, pytest.

---

### Task 1: Huawei MQTT Adapter Contract

**Files:**
- Modify: `deidao/device_adapter.py`
- Test: `test/test_huawei_band_adapter.py`
- Modify: `requirements.txt`
- Modify: `deidao/device_config.json`

- [x] Write failing tests proving that `huawei_band.mqtt` controls broker/topic settings, unsupported values remain missing, blood-pressure estimation is opt-in, and empty messages do not emit fake data.
- [x] Run `.\.venv\Scripts\python.exe -m pytest test/test_huawei_band_adapter.py -q` and verify the new tests fail.
- [x] Normalize nested Huawei configuration before calling `MQTTAdapter.__init__`, derive `gadgetbridge/{device_id}/health`, reject empty payloads, parse optional metrics, and only estimate blood pressure when configured.
- [x] Add `paho-mqtt>=1.6,<3`, update default configuration, and verify all adapter tests pass.

### Task 2: Dashboard Real-Data Path

**Files:**
- Modify: `deidao/dialogs.py`
- Test: `test/test_huawei_band_adapter.py`

- [x] Write failing tests for formatting `NaN` as `--` and for applying one real sample without calling simulator randomization.
- [x] Add class-level device-data/device-error Qt signals and connect them in the dialog constructor.
- [x] Split rendering from simulator updates: MQTT data queues through signals and calls a render-only method; manual refresh still uses simulator mode.
- [x] Run `.\.venv\Scripts\python.exe -m pytest test/test_huawei_band_adapter.py -q` and verify it passes.

### Task 3: Gadgetbridge SQLite Bridge

**Files:**
- Create: `tools/gadgetbridge_mqtt_bridge.py`
- Test: `test/test_gadgetbridge_mqtt_bridge.py`

- [x] Write failing tests against an in-memory SQLite database containing the official Huawei activity table shape and a device battery table shape.
- [x] Implement read-only latest-sample extraction, NaN filtering, UTC ISO timestamps, mtime polling, retained MQTT publishing, and a `--once` mode.
- [x] Run `.\.venv\Scripts\python.exe -m pytest test/test_gadgetbridge_mqtt_bridge.py -q` and verify it passes.

### Task 4: Configuration and Verification

**Files:**
- Modify: `md/HUAWEI_BAND_GUIDE.md`
- Modify: `README.md`

- [x] Replace the false claim that Gadgetbridge has built-in MQTT with the database-bridge architecture and phone export setup.
- [x] Run `.\.venv\Scripts\python.exe -m pytest test/test_device_adapter.py test/test_huawei_band_adapter.py test/test_gadgetbridge_mqtt_bridge.py -q`.
- [x] Run `.\.venv\Scripts\python.exe tools/gadgetbridge_mqtt_bridge.py --help` and inspect the command output.
