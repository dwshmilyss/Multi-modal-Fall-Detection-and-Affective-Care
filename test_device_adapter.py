# -*- coding: utf-8 -*-
"""
设备适配器测试脚本
==================
测试三种模式的设备适配器
"""

import os
import sys
import json
import time
import threading

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deidao.device_adapter import (
    DeviceAdapter, HealthData, create_adapter, load_config, save_config,
    SimulatorAdapter, MQTTAdapter, TCPAdapter
)


def test_health_data():
    """测试 HealthData 数据模型"""
    print("=" * 60)
    print("测试 HealthData 数据模型")
    print("=" * 60)
    
    # 创建数据
    data = HealthData(
        heart_rate=76.0,
        spo2=98.0,
        bp_systolic=120.0,
        bp_diastolic=78.0,
        steps=3420,
        temperature=36.5,
        battery=92.0,
        device_id="test_device"
    )
    
    print(f"创建数据: {data}")
    print(f"转换为字典: {data.to_dict()}")
    
    # 从字典创建
    data_dict = {
        "heart_rate": 80.0,
        "spo2": 97.0,
        "bp_systolic": 122.0,
        "bp_diastolic": 80.0,
        "steps": 3425,
        "temperature": 36.6,
        "battery": 91.0,
        "device_id": "test_device_2",
        "timestamp": "2026-11-07T01:00:00"
    }
    
    data2 = HealthData.from_dict(data_dict)
    print(f"从字典创建: {data2}")
    print(f"心率: {data2.heart_rate}")
    print(f"血氧: {data2.spo2}")
    print()


def test_simulator_adapter():
    """测试模拟器适配器"""
    print("=" * 60)
    print("测试模拟器适配器")
    print("=" * 60)
    
    config = {"mode": "simulator", "update_interval_ms": 1000}
    adapter = SimulatorAdapter(config)
    
    print(f"模式名称: {adapter.mode_name}")
    
    # 连接
    if adapter.connect():
        print("连接成功")
    else:
        print("连接失败")
        return
    
    # 注册回调
    def on_data(data):
        print(f"收到数据: {data}")
    
    adapter.on_data(on_data)
    
    # 启动
    adapter.start()
    print("已启动，等待 3 秒...")
    time.sleep(3)
    
    # 停止
    adapter.stop()
    print("已停止")
    
    # 断开
    adapter.disconnect()
    print("已断开")
    print()


def test_config():
    """测试配置管理"""
    print("=" * 60)
    print("测试配置管理")
    print("=" * 60)
    
    # 加载配置
    config = load_config()
    print(f"加载配置: {json.dumps(config, indent=2, ensure_ascii=False)}")
    
    # 保存配置
    test_config_path = "test_device_config.json"
    save_config(config, test_config_path)
    print(f"配置已保存到: {test_config_path}")
    
    # 重新加载
    config2 = load_config(test_config_path)
    print(f"重新加载配置: {json.dumps(config2, indent=2, ensure_ascii=False)}")
    
    # 清理
    os.remove(test_config_path)
    print()


def test_create_adapter():
    """测试工厂函数"""
    print("=" * 60)
    print("测试工厂函数")
    print("=" * 60)
    
    # 模拟模式
    config_sim = {"mode": "simulator"}
    adapter_sim = create_adapter(config_sim)
    print(f"模拟模式: {adapter_sim.mode_name}")
    
    # MQTT 模式
    config_mqtt = {"mode": "mqtt", "broker": "localhost", "port": 1883}
    adapter_mqtt = create_adapter(config_mqtt)
    print(f"MQTT 模式: {adapter_mqtt.mode_name}")
    
    # TCP 模式
    config_tcp = {"mode": "tcp", "host": "0.0.0.0", "port": 9527}
    adapter_tcp = create_adapter(config_tcp)
    print(f"TCP 模式: {adapter_tcp.mode_name}")
    
    # 未知模式
    try:
        config_unknown = {"mode": "unknown"}
        adapter_unknown = create_adapter(config_unknown)
    except ValueError as e:
        print(f"未知模式异常: {e}")
    
    print()


def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("设备适配器测试")
    print("=" * 60 + "\n")
    
    test_health_data()
    test_config()
    test_create_adapter()
    test_simulator_adapter()
    
    print("=" * 60)
    print("所有测试完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
