# Huawei Demo Fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve simulator-driven health dashboard demos when Huawei Band 9 data is unavailable.

**Architecture:** `HealthDashboardDialog` owns a simulation-fallback timer. In Huawei mode with `fallback_to_simulator=true`, it starts that timer immediately after MQTT starts; the first real Qt-signaled sample stops the timer before applying real data.

**Tech Stack:** Python 3.11, PyQt6 QTimer/signals, pytest.

---

### Task 1: Demo Fallback Behavior

**Files:**
- Modify: `deidao/dialogs.py`
- Modify: `deidao/device_adapter.py`
- Modify: `deidao/device_config.json`
- Test: `test/test_huawei_band_adapter.py`
- Modify: `md/HUAWEI_BAND_GUIDE.md`

- [x] Add failing tests showing fallback starts with a connected fake adapter, real data stops fallback, and disabled fallback does not start on successful connection.
- [x] Implement `fallback_to_simulator`, simulation timer ownership, status text, and real-data takeover.
- [x] Update configuration and guide.
- [x] Run `.\.venv\Scripts\python.exe -m pytest test/test_device_adapter.py test/test_huawei_band_adapter.py test/test_gadgetbridge_mqtt_bridge.py -q`.
