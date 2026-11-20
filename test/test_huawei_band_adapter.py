# -*- coding: utf-8 -*-
import json
import math
import os
import sys
from datetime import datetime
from types import ModuleType
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deidao.device_adapter import HealthData, HuaweiBandAdapter

# dialogs.py imports optional runtime dependencies that are irrelevant to the
# adapter contract; keep these tests usable in a minimal test environment.
for optional_dependency in ("pymysql", "cv2", "numpy"):
    sys.modules.setdefault(optional_dependency, ModuleType(optional_dependency))

from deidao.dialogs import AlarmManager, HealthDashboardDialog, HealthDataSimulator


class FakeSimulator:
    def __init__(self):
        self.heart_rate = math.nan
        self.spo2 = math.nan
        self.bp_systolic = math.nan
        self.bp_diastolic = math.nan
        self.steps = 0
        self.temperature = math.nan
        self.battery = math.nan
        self.history = []
        self.max_history = 43200
        self.update_calls = 0

    def update(self):
        self.update_calls += 1

    def get_status(self, metric, value):
        if metric == "heart_rate":
            return ("暂无数据", "#94A3B8") if math.isnan(value) else (
                ("异常", "#EF4444") if value < 60 or value > 100 else ("正常平稳", "#10B981")
            )
        if metric == "spo2":
            return ("暂无数据", "#94A3B8") if math.isnan(value) else (
                ("偏低", "#F59E0B") if value < 95 else ("优良安全", "#10B981")
            )
        if metric == "bp":
            return ("暂无数据", "#94A3B8") if math.isnan(value[0]) else (
                ("血压异常", "#EF4444") if value[0] > 140 or value[0] < 90 else ("正常范围", "#10B981")
            )
        if metric == "temperature":
            return ("暂无数据", "#94A3B8") if math.isnan(value) else (
                ("发热", "#EF4444") if value > 37.5 else ("无发热", "#10B981")
            )
        if metric == "battery":
            return ("暂无数据", "#94A3B8") if math.isnan(value) else (
                ("电量不足", "#EF4444") if value < 20 else ("续航充足", "#10B981")
            )
        return ("正常", "#10B981")


class FakeLabel:
    def __init__(self):
        self.text = ""
        self.stylesheet = ""

    def setText(self, value):
        self.text = value

    def setStyleSheet(self, value):
        self.stylesheet = value


class FakeTimer:
    def __init__(self):
        self.stopped = 0

    def stop(self):
        self.stopped += 1


class FakeConnectedAdapter:
    mode_name = "华为手环"
    fallback_to_simulator = True

    def __init__(self):
        self.data_callbacks = []
        self.error_callbacks = []
        self.started = False

    def on_data(self, callback):
        self.data_callbacks.append(callback)

    def on_error(self, callback):
        self.error_callbacks.append(callback)

    def connect(self):
        return True

    def start(self):
        self.started = True


def dashboard_without_qt():
    dialog = SimpleNamespace()
    dialog._value_missing = HealthDashboardDialog._value_missing
    dialog._ensure_simulator = lambda: HealthDashboardDialog._ensure_simulator(dialog)
    dialog._render_health_data = lambda: HealthDashboardDialog._render_health_data(dialog)
    dialog.simulator = FakeSimulator()
    dialog.alarm_manager = AlarmManager()
    dialog.card_hr = SimpleNamespace(lbl_val=FakeLabel(), lbl_status=FakeLabel(), accent_color="#0284C7")
    dialog.card_spo2 = SimpleNamespace(lbl_val=FakeLabel(), lbl_status=FakeLabel(), accent_color="#059669")
    dialog.card_bp = SimpleNamespace(lbl_val=FakeLabel(), lbl_status=FakeLabel())
    dialog.card_steps = SimpleNamespace(lbl_val=FakeLabel(), lbl_status=FakeLabel())
    dialog.card_temp = SimpleNamespace(lbl_val=FakeLabel(), lbl_status=FakeLabel(), accent_color="#DC2626")
    dialog.card_battery = SimpleNamespace(lbl_val=FakeLabel(), lbl_status=FakeLabel(), accent_color="#2563EB")
    dialog.update_time_lbl = FakeLabel()
    dialog.update_alarm_stats_display = lambda: None
    return dialog


def config_with(**overrides):
    mqtt = {
        "broker": "192.168.1.20",
        "port": 1884,
        "username": "band",
        "password": "secret",
        "client_id": "sage_client",
        "topic_prefix": "gadgetbridge",
        "device_id": "huawei_band_9",
    }
    mqtt.update(overrides.get("mqtt", {}))
    profile = {"age": 68, "gender": "female", "resting_hr": 64}
    profile.update(overrides.get("user_profile", {}))
    result = {
        "mode": "huawei_band",
        "huawei_band": {"mqtt": mqtt, "user_profile": profile},
    }
    result.update({key: value for key, value in overrides.items() if key not in {"mqtt", "user_profile"}})
    return result


def mqtt_message(payload):
    return SimpleNamespace(payload=json.dumps(payload).encode("utf-8"))


def test_huawei_adapter_uses_nested_mqtt_configuration():
    adapter = HuaweiBandAdapter(config_with())

    assert adapter.config["broker"] == "192.168.1.20"
    assert adapter.config["port"] == 1884
    assert adapter.config["username"] == "band"
    assert adapter.config["password"] == "secret"
    assert adapter.config["client_id"] == "sage_client"
    assert adapter.config["topic_prefix"] == "gadgetbridge"
    assert adapter._device_id == "huawei_band_9"
    assert adapter._topic == "gadgetbridge/huawei_band_9/health"


def test_huawei_adapter_subscribes_to_all_devices_when_device_id_is_empty():
    adapter = HuaweiBandAdapter(config_with(mqtt={"device_id": ""}))

    assert adapter._topic == "gadgetbridge/+/health"


def test_huawei_adapter_preserves_missing_measurements():
    received = []
    adapter = HuaweiBandAdapter(config_with())
    adapter.on_data(received.append)
    adapter._on_message(None, None, mqtt_message({"steps": 3120}))

    assert len(received) == 1
    data = received[0]
    assert data.steps == 3120
    assert math.isnan(data.heart_rate)
    assert math.isnan(data.spo2)
    assert math.isnan(data.temperature)
    assert math.isnan(data.battery)
    assert math.isnan(data.bp_systolic)
    assert math.isnan(data.bp_diastolic)
    assert data.device_id == "huawei_band_9"


def test_huawei_adapter_accepts_supported_measurements_without_faking_blood_pressure():
    received = []
    adapter = HuaweiBandAdapter(config_with())
    adapter.on_data(received.append)
    adapter._on_message(None, None, mqtt_message({
        "device_id": "huawei_band_9",
        "timestamp": "2026-11-19T10:30:00",
        "heart_rate": 82,
        "spo2": 97,
        "steps": 4100,
        "battery": 76,
    }))

    data = received[0]
    assert data.heart_rate == 82
    assert data.spo2 == 97
    assert data.steps == 4100
    assert data.battery == 76
    assert data.timestamp == datetime(2026, 11, 19, 10, 30)
    assert data.device_id == "huawei_band_9"
    assert data.raw_data["heart_rate"] == 82


def test_huawei_adapter_rejects_payload_without_supported_measurements():
    received = []
    errors = []
    adapter = HuaweiBandAdapter(config_with())
    adapter.on_data(received.append)
    adapter.on_error(errors.append)
    adapter._on_message(None, None, mqtt_message({"device_id": "huawei_band_9"}))

    assert received == []
    assert len(errors) == 1
    assert "没有可用的健康指标" in errors[0]


def test_huawei_adapter_estimates_blood_pressure_only_when_enabled():
    received = []
    adapter = HuaweiBandAdapter(config_with(estimate_blood_pressure=True))
    adapter.on_data(received.append)
    adapter._on_message(None, None, mqtt_message({"heart_rate": 74, "steps": 10}))

    assert received[0].bp_systolic == 130.0
    assert received[0].bp_diastolic == 82.0


def test_dashboard_applies_real_data_without_simulator_randomization():
    dialog = dashboard_without_qt()
    sample = HealthData(
        heart_rate=82,
        spo2=97,
            bp_systolic=math.nan,
            bp_diastolic=math.nan,
        steps=4100,
            temperature=math.nan,
        battery=76,
        device_id="huawei_band_9",
    )

    HealthDashboardDialog._apply_health_data(dialog, sample)

    assert dialog.simulator.update_calls == 0
    assert dialog.simulator.heart_rate == 82
    assert dialog.simulator.spo2 == 97
    assert dialog.simulator.steps == 4100
    assert dialog.simulator.battery == 76
    assert dialog.simulator.history[-1][1:] == (82, 97, math.nan, math.nan, 4100, math.nan, 76)
    assert dialog.card_hr.lbl_val.text == "82 bpm"
    assert dialog.card_spo2.lbl_val.text == "97.0 %"


def test_dashboard_renders_missing_device_values_as_unavailable():
    dialog = dashboard_without_qt()
    sample = HealthData(
        heart_rate=math.nan,
        spo2=math.nan,
        bp_systolic=math.nan,
        bp_diastolic=math.nan,
        steps=3120,
        temperature=math.nan,
        battery=math.nan,
        device_id="huawei_band_9",
    )

    HealthDashboardDialog._apply_health_data(dialog, sample)

    assert dialog.card_hr.lbl_val.text == "-- bpm"
    assert dialog.card_hr.lbl_status.text == "状态：暂无数据"
    assert dialog.card_spo2.lbl_val.text == "-- %"
    assert dialog.card_bp.lbl_val.text == "-- / -- mmHg"
    assert dialog.card_temp.lbl_val.text == "-- ℃"
    assert dialog.card_battery.lbl_val.text == "-- %"


def huawei_fallback_dashboard(monkeypatch, enabled=True):
    dialog = dashboard_without_qt()
    config = config_with()
    config["huawei_band"]["fallback_to_simulator"] = enabled
    dialog.device_config = config
    dialog.device_adapter = FakeConnectedAdapter()
    dialog.device_adapter.fallback_to_simulator = enabled
    dialog.device_status_lbl = FakeLabel()
    timer_starts = []
    dialog._start_timer = lambda mark_fallback=True: timer_starts.append(mark_fallback)
    return dialog, timer_starts


def test_huawei_mode_starts_simulator_fallback_while_waiting_for_real_data(monkeypatch):
    dialog, timer_starts = huawei_fallback_dashboard(monkeypatch, enabled=True)

    HealthDashboardDialog.start_data_source(dialog)

    assert dialog.device_adapter.started is True
    assert timer_starts == [True]
    assert "模拟演示等待真实数据" in dialog.device_status_lbl.text


def test_demo_fallback_creates_simulator_when_huawei_mode_starts_without_one():
    dialog = dashboard_without_qt()
    dialog.simulator = None

    HealthDashboardDialog._ensure_simulator(dialog)

    assert isinstance(dialog.simulator, HealthDataSimulator)


def test_first_real_huawei_message_takes_over_from_simulation_fallback(monkeypatch):
    dialog, _ = huawei_fallback_dashboard(monkeypatch, enabled=True)
    timer = FakeTimer()
    dialog.timer = timer
    dialog.simulation_fallback_active = True
    applied = []
    dialog._apply_health_data = lambda data: applied.append(data)
    sample = HealthData(heart_rate=82, device_id="huawei_band_9")

    HealthDashboardDialog._on_device_data(dialog, sample)

    assert timer.stopped == 1
    assert dialog.simulation_fallback_active is False
    assert applied == [sample]
    assert "真实数据" in dialog.device_status_lbl.text


def test_huawei_demo_fallback_can_be_disabled(monkeypatch):
    dialog, timer_starts = huawei_fallback_dashboard(monkeypatch, enabled=False)

    HealthDashboardDialog.start_data_source(dialog)

    assert dialog.device_adapter.started is True
    assert timer_starts == []
    assert "已连接" in dialog.device_status_lbl.text
    assert "模拟演示" not in dialog.device_status_lbl.text


def install_fake_mqtt_module(monkeypatch, client_class):
    paho_package = ModuleType("paho")
    mqtt_package = ModuleType("paho.mqtt")
    client_module = ModuleType("paho.mqtt.client")
    client_module.Client = client_class
    client_module.MQTTv311 = object()
    mqtt_package.client = client_module
    paho_package.mqtt = mqtt_package
    monkeypatch.setitem(sys.modules, "paho", paho_package)
    monkeypatch.setitem(sys.modules, "paho.mqtt", mqtt_package)
    monkeypatch.setitem(sys.modules, "paho.mqtt.client", client_module)


def test_huawei_adapter_waits_for_broker_connection(monkeypatch):
    class FakeMqttClient:
        subscriptions = []

        def __init__(self, *args, **kwargs):
            self.on_connect = None
            self.on_message = None
            self.on_disconnect = None
            self.commands = []

        def username_pw_set(self, username, password):
            self.commands.append(("username_pw_set", username, password))

        def reconnect_delay_set(self, **kwargs):
            self.commands.append(("reconnect_delay_set", kwargs))

        def connect_async(self, broker, port, keepalive):
            self.commands.append(("connect_async", broker, port, keepalive))
            self.on_connect(self, None, {}, 0)

        def loop_start(self):
            self.commands.append("loop_start")

        def subscribe(self, topic):
            self.subscriptions.append(topic)

        def disconnect(self):
            self.commands.append("disconnect")

        def loop_stop(self):
            self.commands.append("loop_stop")

    install_fake_mqtt_module(monkeypatch, FakeMqttClient)
    adapter = HuaweiBandAdapter(config_with(mqtt={"connect_timeout_s": 0.2}))

    assert adapter.connect() is True
    assert adapter.connection_status == "connected"
    assert FakeMqttClient.subscriptions == ["gadgetbridge/huawei_band_9/health"]


def test_huawei_adapter_reports_timeout_when_broker_never_connects(monkeypatch):
    class FakeMqttClient:
        def __init__(self, *args, **kwargs):
            self.on_connect = None
            self.on_message = None
            self.on_disconnect = None

        def username_pw_set(self, username, password):
            pass

        def reconnect_delay_set(self, **kwargs):
            pass

        def connect_async(self, broker, port, keepalive):
            pass

        def loop_start(self):
            pass

        def disconnect(self):
            pass

        def loop_stop(self):
            pass

    install_fake_mqtt_module(monkeypatch, FakeMqttClient)
    errors = []
    adapter = HuaweiBandAdapter(config_with(mqtt={"connect_timeout_s": 0.01}))
    adapter.on_error(errors.append)

    assert adapter.connect() is False
    assert adapter.connection_status == "error"
    assert errors and "连接超时" in errors[0]


def test_huawei_adapter_cleans_up_after_broker_rejects_connection(monkeypatch):
    class FakeMqttClient:
        instances = []

        def __init__(self, *args, **kwargs):
            self.on_connect = None
            self.on_message = None
            self.on_disconnect = None
            self.commands = []
            FakeMqttClient.instances.append(self)

        def username_pw_set(self, username, password):
            pass

        def reconnect_delay_set(self, **kwargs):
            pass

        def connect_async(self, broker, port, keepalive):
            self.on_connect(self, None, {}, 5)

        def loop_start(self):
            self.commands.append("loop_start")

        def disconnect(self):
            self.commands.append("disconnect")

        def loop_stop(self):
            self.commands.append("loop_stop")

    install_fake_mqtt_module(monkeypatch, FakeMqttClient)
    errors = []
    adapter = HuaweiBandAdapter(config_with(mqtt={"connect_timeout_s": 0.2}))
    adapter.on_error(errors.append)

    assert adapter.connect() is False
    assert adapter.connection_status == "error"
    assert adapter._mqtt_client is None
    assert FakeMqttClient.instances[-1].commands[-2:] == ["disconnect", "loop_stop"]
    assert errors and "返回码: 5" in errors[0]
