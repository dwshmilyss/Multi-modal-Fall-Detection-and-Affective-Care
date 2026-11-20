# -*- coding: utf-8 -*-
"""Publish the latest Huawei samples from a Gadgetbridge SQLite export to MQTT."""

import argparse
import json
import math
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


ACTIVITY_TABLE = "HUAWEI_ACTIVITY_SAMPLE"


def _missing(value: Any) -> bool:
    try:
        return value is None or math.isnan(float(value))
    except (TypeError, ValueError):
        return True


def _open_read_only(database_path: str) -> sqlite3.Connection:
    path = Path(database_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Gadgetbridge 数据库不存在: {path}")
    return sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True, timeout=3)


def _table_exists(connection: sqlite3.Connection, table: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND upper(NAME)=upper(?)",
        (table,),
    ).fetchone()
    return row is not None


def _columns(connection: sqlite3.Connection, table: str):
    return {row[1].upper() for row in connection.execute(f'PRAGMA table_info("{table}")')}


def _find_device_id(connection: sqlite3.Connection, device_query: str) -> Optional[int]:
    if not device_query or not _table_exists(connection, "DEVICE"):
        return None

    columns = _columns(connection, "DEVICE")
    id_column = "_ID" if "_ID" in columns else "ID"
    conditions = []
    parameters = []
    for column in ("NAME", "MANUFACTURER", "MODEL", "ALIAS", "IDENTIFIER"):
        if column in columns:
            conditions.append(f'lower("{column}") LIKE ?')
            parameters.append(f"%{device_query.lower()}%")
    if not conditions:
        return None

    row = connection.execute(
        f'SELECT "{id_column}" FROM DEVICE WHERE {" OR ".join(conditions)} ORDER BY "{id_column}" LIMIT 1',
        parameters,
    ).fetchone()
    return int(row[0]) if row else None


def _latest_activity_value(
    connection: sqlite3.Connection,
    value_column: str,
    device_id: Optional[int],
):
    query = f"""
        SELECT "TIMESTAMP", "{value_column}"
        FROM "{ACTIVITY_TABLE}"
        WHERE "{value_column}" IS NOT NULL
          AND "{value_column}" != -1
          AND "TIMESTAMP" <= "OTHER_TIMESTAMP"
    """
    parameters = []
    if device_id is not None:
        query += ' AND "DEVICE_ID" = ?'
        parameters.append(device_id)
    query += ' ORDER BY "TIMESTAMP" DESC LIMIT 1'

    row = connection.execute(query, parameters).fetchone()
    if not row:
        return math.nan, math.nan
    timestamp, value = int(row[0]), row[1]
    if value_column == "HEART_RATE":
        value = int(value) & 0xFF
    return float(value), float(timestamp)


def _today_steps(connection: sqlite3.Connection, device_id: Optional[int]):
    midnight = datetime.now().replace(
        hour=0, minute=0, second=0, microsecond=0
    ).timestamp()
    query = f"""
        SELECT SUM("STEPS"), MAX("TIMESTAMP")
        FROM "{ACTIVITY_TABLE}"
        WHERE "STEPS" IS NOT NULL
          AND "STEPS" >= 0
          AND "TIMESTAMP" >= ?
          AND "TIMESTAMP" <= "OTHER_TIMESTAMP"
    """
    parameters = [int(midnight)]
    if device_id is not None:
        query += ' AND "DEVICE_ID" = ?'
        parameters.append(device_id)

    row = connection.execute(query, parameters).fetchone()
    if not row or row[0] is None:
        return math.nan, math.nan
    return float(row[0]), float(row[1])


def _latest_battery(connection: sqlite3.Connection, device_id: Optional[int]):
    if _table_exists(connection, "BATTERY_LEVEL"):
        columns = _columns(connection, "BATTERY_LEVEL")
        if "LEVEL" in columns:
            timestamp_column = "TIMESTAMP" if "TIMESTAMP" in columns else "_id"
            query = f'SELECT "{timestamp_column}", "LEVEL" FROM BATTERY_LEVEL WHERE "LEVEL" >= 0'
            parameters = []
            if device_id is not None and "DEVICE_ID" in columns:
                query += ' AND "DEVICE_ID" = ?'
                parameters.append(device_id)
            query += f' ORDER BY "{timestamp_column}" DESC LIMIT 1'
            row = connection.execute(query, parameters).fetchone()
            if row:
                return float(row[1]), float(row[0])

    if _table_exists(connection, "DEVICE_ATTRIBUTES"):
        columns = _columns(connection, "DEVICE_ATTRIBUTES")
        if {"DEVICE_ID", "KEY", "VALUE"}.issubset(columns):
            query = """
                SELECT "VALUE" FROM DEVICE_ATTRIBUTES
                WHERE lower("KEY") IN ('battery_level', 'battery')
            """
            parameters = []
            if device_id is not None:
                query += ' AND "DEVICE_ID" = ?'
                parameters.append(device_id)
            query += ' ORDER BY rowid DESC LIMIT 1'
            row = connection.execute(query, parameters).fetchone()
            if row:
                try:
                    return float(row[0]), math.nan
                except (TypeError, ValueError):
                    pass

    return math.nan, math.nan


def load_latest_health_data(
    database_path: str,
    device_query: str = "huawei band 9",
    device_id: str = "huawei_band_9",
) -> Optional[Dict[str, Any]]:
    """Read the latest supported samples without modifying Gadgetbridge's DB."""
    connection = _open_read_only(database_path)
    try:
        if not _table_exists(connection, ACTIVITY_TABLE):
            raise ValueError(f"数据库缺少 {ACTIVITY_TABLE} 表，请确认设备已同步数据")

        internal_device_id = _find_device_id(connection, device_query)
        heart_rate, hr_timestamp = _latest_activity_value(
            connection, "HEART_RATE", internal_device_id
        )
        spo2, spo2_timestamp = _latest_activity_value(
            connection, "SPO", internal_device_id
        )
        steps, steps_timestamp = _today_steps(connection, internal_device_id)
        battery, battery_timestamp = _latest_battery(connection, internal_device_id)
    finally:
        connection.close()

    supported = (heart_rate, spo2, steps, battery)
    if all(_missing(value) for value in supported):
        return None

    timestamps = [value for value in (hr_timestamp, spo2_timestamp, steps_timestamp, battery_timestamp) if not _missing(value)]
    return {
        "device_id": device_id,
        "timestamp_unix": int(max(timestamps)),
        "heart_rate": heart_rate,
        "spo2": spo2,
        "steps": steps,
        "battery": battery,
    }


def build_payload(sample: Dict[str, Any]) -> Dict[str, Any]:
    payload = {
        "device_id": str(sample.get("device_id", "huawei_band_9")),
        "timestamp": datetime.fromtimestamp(
            int(sample["timestamp_unix"]), timezone.utc
        ).isoformat(),
    }
    for key in ("heart_rate", "spo2", "steps", "battery"):
        value = sample.get(key)
        if not _missing(value):
            payload[key] = int(value)
    payload["json"] = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return payload


def publish_once(
    database_path: str,
    publisher,
    device_query: str = "huawei band 9",
    device_id: str = "huawei_band_9",
    topic_prefix: str = "gadgetbridge",
) -> bool:
    sample = load_latest_health_data(
        database_path, device_query=device_query, device_id=device_id
    )
    if sample is None:
        return False
    payload = build_payload(sample)
    publisher.publish(f"{topic_prefix}/{device_id}/health", payload["json"], retain=True)
    return True


def follow_database(
    database_path: str,
    publisher,
    poll_seconds: float = 5.0,
    device_query: str = "huawei band 9",
    device_id: str = "huawei_band_9",
    topic_prefix: str = "gadgetbridge",
):
    last_mtime = None
    last_json = None
    while True:
        try:
            current_mtime = os.stat(database_path).st_mtime_ns
            if current_mtime != last_mtime:
                sample = load_latest_health_data(
                    database_path, device_query=device_query, device_id=device_id
                )
                if sample is not None:
                    payload = build_payload(sample)
                    if payload["json"] != last_json:
                        publisher.publish(
                            f"{topic_prefix}/{device_id}/health",
                            payload["json"],
                            retain=True,
                        )
                        last_json = payload["json"]
                last_mtime = current_mtime
        except (OSError, sqlite3.Error, ValueError) as error:
            print(f"等待 Gadgetbridge 数据库: {error}")
        time.sleep(poll_seconds)


class DryRunPublisher:
    def publish(self, topic, payload, retain=True):
        print(f"{topic} retain={retain} {payload}")


class MqttPublisher:
    def __init__(self, broker, port, username=None, password=None):
        import paho.mqtt.client as mqtt

        self.client = mqtt.Client(client_id="sage_gadgetbridge_bridge", protocol=mqtt.MQTTv311)
        if username:
            self.client.username_pw_set(username, password or "")
        self.client.connect(broker, port, keepalive=30)
        self.client.loop_start()

    def publish(self, topic, payload, retain=True):
        self.client.publish(topic, payload, qos=1, retain=retain).wait_for_publish(5)

    def close(self):
        self.client.disconnect()
        self.client.loop_stop()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, help="Gadgetbridge.db 路径")
    parser.add_argument("--broker", default="localhost")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--username", default="")
    parser.add_argument("--password", default="")
    parser.add_argument("--device-query", default="huawei band 9")
    parser.add_argument("--device-id", default="huawei_band_9")
    parser.add_argument("--topic-prefix", default="gadgetbridge")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--once", action="store_true", help="读取并发布一次后退出")
    parser.add_argument("--dry-run", action="store_true", help="只打印 MQTT 消息，不连接 Broker")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    publisher = DryRunPublisher() if args.dry_run else MqttPublisher(
        args.broker, args.port, args.username, args.password
    )
    try:
        if args.once:
            if not publish_once(
                args.db,
                publisher,
                args.device_query,
                args.device_id,
                args.topic_prefix,
            ):
                print("数据库中没有可发布的手环数据")
            return
        follow_database(
            args.db,
            publisher,
            args.poll_seconds,
            args.device_query,
            args.device_id,
            args.topic_prefix,
        )
    finally:
        if hasattr(publisher, "close"):
            publisher.close()


if __name__ == "__main__":
    main()
