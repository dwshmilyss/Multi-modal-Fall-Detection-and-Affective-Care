# -*- coding: utf-8 -*-
import json
import math
import os
import sqlite3
import sys
import tempfile
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.gadgetbridge_mqtt_bridge import (
    build_payload,
    load_latest_health_data,
    publish_once,
)


class FakePublisher:
    def __init__(self):
        self.published = []

    def publish(self, topic, payload, retain=True):
        self.published.append((topic, payload, retain))


def create_database():
    handle = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    handle.close()
    path = handle.name
    connection = sqlite3.connect(path)
    connection.execute("""
        CREATE TABLE DEVICE (
            _id INTEGER PRIMARY KEY,
            NAME TEXT,
            IDENTIFIER TEXT
        )
    """)
    connection.execute("""
        CREATE TABLE HUAWEI_ACTIVITY_SAMPLE (
            TIMESTAMP INTEGER,
            DEVICE_ID INTEGER,
            USER_ID INTEGER,
            OTHER_TIMESTAMP INTEGER,
            SOURCE INTEGER,
            RAW_KIND INTEGER,
            RAW_INTENSITY INTEGER,
            STEPS INTEGER,
            CALORIES INTEGER,
            DISTANCE REAL,
            SPO INTEGER,
            HEART_RATE INTEGER,
            RESTING_HEART_RATE INTEGER,
            PRIMARY KEY(TIMESTAMP, DEVICE_ID, OTHER_TIMESTAMP)
        )
    """)
    connection.execute("""
        CREATE TABLE BATTERY_LEVEL (
            TIMESTAMP INTEGER,
            DEVICE_ID INTEGER,
            LEVEL INTEGER,
            PRIMARY KEY(TIMESTAMP, DEVICE_ID)
        )
    """)
    connection.execute(
        "INSERT INTO DEVICE(_id, NAME, IDENTIFIER) VALUES(1, 'huawei band 9-XXXX', 'AA:BB:CC:DD:EE:FF')"
    )
    connection.commit()
    return path, connection


def insert_activity(connection, timestamp, hr=-1, spo=-1, steps=-1):
    connection.execute(
        """
        INSERT INTO HUAWEI_ACTIVITY_SAMPLE(
            TIMESTAMP, DEVICE_ID, USER_ID, OTHER_TIMESTAMP, SOURCE, RAW_KIND,
            RAW_INTENSITY, STEPS, CALORIES, DISTANCE, SPO, HEART_RATE,
            RESTING_HEART_RATE
        ) VALUES(?, 1, 1, ?, 0, -1, 0, ?, -1, -1, ?, ?, -1)
        """,
        (timestamp, timestamp + 60, steps, spo, hr),
    )


def test_load_latest_health_data_from_gadgetbridge_schema():
    path, connection = create_database()
    try:
        midnight = int(
            datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        )
        insert_activity(connection, midnight + 60, hr=78, spo=96, steps=4)
        insert_activity(connection, midnight + 120, spo=98, steps=7)
        insert_activity(connection, midnight + 180, hr=82)
        connection.execute(
            "INSERT INTO BATTERY_LEVEL(TIMESTAMP, DEVICE_ID, LEVEL) VALUES(?, 1, 76)",
            (midnight + 180,),
        )
        connection.commit()

        sample = load_latest_health_data(path, device_query="huawei band 9")

        assert sample["heart_rate"] == 82
        assert sample["spo2"] == 98
        assert sample["steps"] == 11
        assert sample["battery"] == 76
        assert sample["device_id"] == "huawei_band_9"
        assert datetime.fromtimestamp(sample["timestamp_unix"], timezone.utc).isoformat().endswith("+00:00")
    finally:
        connection.close()
        os.remove(path)


def test_build_payload_omits_missing_metrics_and_serializes_json():
    payload = build_payload({
        "device_id": "huawei_band_9",
        "timestamp_unix": 1763568000,
        "heart_rate": 82,
        "spo2": math.nan,
        "steps": 4100,
        "battery": math.nan,
    })

    assert payload["device_id"] == "huawei_band_9"
    assert payload["timestamp"] == "2025-11-19T16:00:00+00:00"
    assert payload["heart_rate"] == 82
    assert payload["steps"] == 4100
    assert "spo2" not in payload
    assert "battery" not in payload
    assert json.loads(payload["json"]) == {
        "device_id": "huawei_band_9",
        "timestamp": "2025-11-19T16:00:00+00:00",
        "heart_rate": 82,
        "steps": 4100,
    }


def test_load_returns_none_when_no_supported_values_exist():
    path, connection = create_database()
    try:
        insert_activity(connection, int(datetime.now().timestamp()), hr=-1, spo=-1, steps=-1)
        connection.commit()

        assert load_latest_health_data(path, device_query="huawei band 9") is None
    finally:
        connection.close()
        os.remove(path)


def test_publish_once_uses_configured_topic_and_retained_message():
    path, connection = create_database()
    try:
        now = int(datetime.now().timestamp())
        insert_activity(connection, now, hr=82, spo=98, steps=10)
        connection.commit()
        publisher = FakePublisher()

        result = publish_once(
            path,
            publisher,
            device_query="huawei band 9",
            device_id="bedroom_band",
            topic_prefix="sage",
        )

        assert result is True
        topic, json_payload, retain = publisher.published[0]
        assert topic == "sage/bedroom_band/health"
        assert json.loads(json_payload)["heart_rate"] == 82
        assert retain is True
    finally:
        connection.close()
        os.remove(path)
