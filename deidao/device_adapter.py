# -*- coding: utf-8 -*-
"""
设备数据适配器框架
==================
提供统一的设备数据接口，支持三种模式：
1. 模拟模式 (SimulatorAdapter) - 使用 HealthDataSimulator 生成模拟数据
2. MQTT 模式 (MQTTAdapter) - 通过 MQTT 协议接入真实智能手环/传感器
3. TCP 模式 (TCPAdapter) - 通过 TCP Socket 接入自定义设备

设计原则：
- 所有适配器实现相同的接口（DeviceAdapter 基类）
- 通过配置文件切换模式，无需修改业务代码
- 真实设备模式与模拟模式无缝切换
"""

import json
import os
import time
import threading
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, Dict, Any, Callable, List, Tuple

logger = logging.getLogger(__name__)


# ==================== 血压估算算法 ====================

class BloodPressureEstimator:
    """
    基于心率变异性 (HRV) 的血压估算算法
    
    原理：
    - 收缩压 (SBP) 与心率呈正相关（心率越高，收缩压越高）
    - 舒张压 (DBP) 与心率呈弱正相关
    - 年龄也是影响血压的重要因素
    
    注意：这是估算值，仅供参考，不能替代医疗设备测量
    """
    
    @staticmethod
    def estimate(
        heart_rate: float,
        age: float = 65.0,
        gender: str = "male",
        resting_hr: float = 70.0
    ) -> Tuple[float, float]:
        """
        基于心率估算血压
        
        Args:
            heart_rate: 当前心率 (bpm)
            age: 年龄 (岁)
            gender: 性别 ("male" / "female")
            resting_hr: 静息心率 (bpm)
        
        Returns:
            (收缩压, 舒张压) 单位：mmHg
        """
        # 基础血压值（根据年龄和性别）
        if age < 40:
            base_sbp = 110 if gender == "male" else 105
            base_dbp = 70
        elif age < 60:
            base_sbp = 120 if gender == "male" else 115
            base_dbp = 75
        else:
            base_sbp = 130 if gender == "male" else 125
            base_dbp = 80
        
        # 心率对收缩压的影响（心率每增加 10 bpm，收缩压约增加 5 mmHg）
        hr_diff = heart_rate - resting_hr
        sbp_adjustment = hr_diff * 0.5
        
        # 心率对舒张压的影响（较弱，心率每增加 10 bpm，舒张压约增加 2 mmHg）
        dbp_adjustment = hr_diff * 0.2
        
        # 计算最终血压
        sbp = base_sbp + sbp_adjustment
        dbp = base_dbp + dbp_adjustment
        
        # 限制在合理范围内
        sbp = max(90, min(180, sbp))
        dbp = max(60, min(110, dbp))
        
        # 确保收缩压 > 舒张压
        if sbp <= dbp:
            sbp = dbp + 30
        
        return round(sbp, 1), round(dbp, 1)
    
    @staticmethod
    def estimate_from_history(
        heart_rates: List[float],
        age: float = 65.0,
        gender: str = "male"
    ) -> Tuple[float, float]:
        """
        基于心率历史数据估算血压（更稳定）
        
        Args:
            heart_rates: 心率历史数据列表
            age: 年龄
            gender: 性别
        
        Returns:
            (收缩压, 舒张压)
        """
        if not heart_rates:
            return 120.0, 80.0
        
        # 使用平均心率
        avg_hr = sum(heart_rates) / len(heart_rates)
        
        # 使用静息心率（取最小值）
        resting_hr = min(heart_rates)
        
        return BloodPressureEstimator.estimate(avg_hr, age, gender, resting_hr)


# ==================== 数据模型 ====================

class HealthData:
    """健康数据结构（所有适配器统一返回此格式）"""
    
    def __init__(
        self,
        heart_rate: float = 0.0,
        spo2: float = 0.0,
        bp_systolic: float = 0.0,
        bp_diastolic: float = 0.0,
        steps: int = 0,
        temperature: float = 0.0,
        battery: float = 0.0,
        timestamp: Optional[datetime] = None,
        device_id: str = "unknown",
        raw_data: Optional[Dict] = None,
    ):
        self.heart_rate = heart_rate
        self.spo2 = spo2
        self.bp_systolic = bp_systolic
        self.bp_diastolic = bp_diastolic
        self.steps = steps
        self.temperature = temperature
        self.battery = battery
        self.timestamp = timestamp or datetime.now()
        self.device_id = device_id
        self.raw_data = raw_data or {}  # 保留原始数据供调试
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "heart_rate": self.heart_rate,
            "spo2": self.spo2,
            "bp_systolic": self.bp_systolic,
            "bp_diastolic": self.bp_diastolic,
            "steps": self.steps,
            "temperature": self.temperature,
            "battery": self.battery,
            "timestamp": self.timestamp.isoformat(),
            "device_id": self.device_id,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'HealthData':
        ts = data.get("timestamp")
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        return cls(
            heart_rate=float(data.get("heart_rate", 0)),
            spo2=float(data.get("spo2", 0)),
            bp_systolic=float(data.get("bp_systolic", 0)),
            bp_diastolic=float(data.get("bp_diastolic", 0)),
            steps=int(data.get("steps", 0)),
            temperature=float(data.get("temperature", 0)),
            battery=float(data.get("battery", 0)),
            timestamp=ts,
            device_id=str(data.get("device_id", "unknown")),
            raw_data=data,
        )
    
    def __repr__(self):
        return (f"HealthData(hr={self.heart_rate:.0f}, spo2={self.spo2:.1f}, "
                f"bp={self.bp_systolic:.0f}/{self.bp_diastolic:.0f}, "
                f"temp={self.temperature:.1f}, batt={self.battery:.1f}%)")


# ==================== 基类 ====================

class DeviceAdapter(ABC):
    """设备数据适配器基类 - 所有适配器必须实现此接口"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._running = False
        self._callbacks: List[Callable[[HealthData], None]] = []
        self._error_callbacks: List[Callable[[str], None]] = []
        self._last_data: Optional[HealthData] = None
        self._connection_status = "disconnected"  # disconnected / connecting / connected / error
    
    @property
    def connection_status(self) -> str:
        return self._connection_status
    
    @property
    def last_data(self) -> Optional[HealthData]:
        return self._last_data
    
    def on_data(self, callback: Callable[[HealthData], None]):
        """注册数据回调"""
        self._callbacks.append(callback)
    
    def on_error(self, callback: Callable[[str], None]):
        """注册错误回调"""
        self._error_callbacks.append(callback)
    
    def _emit_data(self, data: HealthData):
        """向所有回调发送数据"""
        self._last_data = data
        for cb in self._callbacks:
            try:
                cb(data)
            except Exception as e:
                logger.error(f"数据回调异常: {e}")
    
    def _emit_error(self, message: str):
        """向所有错误回调发送错误"""
        for cb in self._error_callbacks:
            try:
                cb(message)
            except Exception:
                pass
    
    @abstractmethod
    def connect(self) -> bool:
        """连接到设备，返回是否成功"""
        pass
    
    @abstractmethod
    def disconnect(self):
        """断开连接"""
        pass
    
    @abstractmethod
    def start(self):
        """开始接收数据"""
        pass
    
    @abstractmethod
    def stop(self):
        """停止接收数据"""
        pass
    
    @property
    @abstractmethod
    def mode_name(self) -> str:
        """返回模式名称（用于 UI 显示）"""
        pass


# ==================== 模拟模式适配器 ====================

class SimulatorAdapter(DeviceAdapter):
    """模拟数据适配器 - 使用 HealthDataSimulator 生成数据"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._simulator = None
        self._timer = None
    
    @property
    def mode_name(self) -> str:
        return "模拟模式"
    
    def connect(self) -> bool:
        """初始化模拟器"""
        try:
            # 延迟导入避免循环依赖
            from deidao.dialogs import HealthDataSimulator
            self._simulator = HealthDataSimulator()
            self._connection_status = "connected"
            logger.info("模拟器已初始化")
            return True
        except Exception as e:
            self._connection_status = "error"
            self._emit_error(f"模拟器初始化失败: {e}")
            return False
    
    def disconnect(self):
        self.stop()
        self._simulator = None
        self._connection_status = "disconnected"
    
    def start(self):
        """开始定时生成数据"""
        if self._simulator is None:
            return
        self._running = True
        interval = self.config.get("update_interval_ms", 2000)
        
        def _tick():
            while self._running:
                try:
                    self._simulator.update()
                    data = HealthData(
                        heart_rate=self._simulator.heart_rate,
                        spo2=self._simulator.spo2,
                        bp_systolic=self._simulator.bp_systolic,
                        bp_diastolic=self._simulator.bp_diastolic,
                        steps=self._simulator.steps,
                        temperature=self._simulator.temperature,
                        battery=self._simulator.battery,
                        device_id="simulator",
                    )
                    self._emit_data(data)
                except Exception as e:
                    self._emit_error(f"模拟数据生成异常: {e}")
                time.sleep(interval / 1000.0)
        
        self._timer = threading.Thread(target=_tick, daemon=True)
        self._timer.start()
    
    def stop(self):
        self._running = False
        if self._timer:
            self._timer.join(timeout=3)
            self._timer = None


# ==================== MQTT 模式适配器 ====================

class MQTTAdapter(DeviceAdapter):
    """
    MQTT 设备适配器 - 通过 MQTT 协议接入真实智能手环/传感器
    
    数据格式约定（JSON）：
    {
        "device_id": "band_001",
        "timestamp": "2026-11-07T01:00:00",
        "heart_rate": 76,
        "spo2": 98,
        "bp_systolic": 120,
        "bp_diastolic": 78,
        "steps": 3420,
        "temperature": 36.5,
        "battery": 92
    }
    
    订阅主题：{topic_prefix}/{device_id}/health
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._client = None
        self._topic = config.get("topic", "health/device/+")
        self._device_id = config.get("device_id", "")
        self._mqtt_client = None
    
    @property
    def mode_name(self) -> str:
        return "MQTT 模式"
    
    def connect(self) -> bool:
        """连接到 MQTT Broker"""
        try:
            import paho.mqtt.client as mqtt
        except ImportError:
            self._connection_status = "error"
            self._emit_error("MQTT 适配器需要安装 paho-mqtt: pip install paho-mqtt")
            return False
        
        try:
            broker = self.config.get("broker", "localhost")
            port = self.config.get("port", 1883)
            username = self.config.get("username", "")
            password = self.config.get("password", "")
            client_id = self.config.get("client_id", "health_monitor_client")
            
            self._mqtt_client = mqtt.Client(
                client_id=client_id,
                protocol=mqtt.MQTTv311,
            )
            
            if username:
                self._mqtt_client.username_pw_set(username, password)
            
            self._mqtt_client.on_connect = self._on_connect
            self._mqtt_client.on_message = self._on_message
            self._mqtt_client.on_disconnect = self._on_disconnect
            
            self._connection_status = "connecting"
            self._mqtt_client.connect(broker, port, keepalive=60)
            self._mqtt_client.loop_start()
            
            return True
        except Exception as e:
            self._connection_status = "error"
            self._emit_error(f"MQTT 连接失败: {e}")
            return False
    
    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self._connection_status = "connected"
            topic = self._topic
            if self._device_id:
                topic = f"{self.config.get('topic_prefix', 'health')}/{self._device_id}/health"
            client.subscribe(topic)
            logger.info(f"MQTT 已连接，订阅主题: {topic}")
        else:
            self._connection_status = "error"
            self._emit_error(f"MQTT 连接被拒绝，返回码: {rc}")
    
    def _on_message(self, client, userdata, msg):
        """处理收到的 MQTT 消息"""
        try:
            payload = json.loads(msg.payload.decode('utf-8'))
            data = HealthData.from_dict(payload)
            self._emit_data(data)
        except json.JSONDecodeError as e:
            self._emit_error(f"MQTT 消息解析失败: {e}")
        except Exception as e:
            self._emit_error(f"MQTT 消息处理异常: {e}")
    
    def _on_disconnect(self, client, userdata, rc):
        if rc != 0:
            self._connection_status = "error"
            self._emit_error(f"MQTT 意外断开，返回码: {rc}")
        else:
            self._connection_status = "disconnected"
    
    def disconnect(self):
        self.stop()
        if self._mqtt_client:
            self._mqtt_client.loop_stop()
            self._mqtt_client.disconnect()
            self._mqtt_client = None
        self._connection_status = "disconnected"
    
    def start(self):
        """MQTT 模式下，start 仅标记为运行中（数据由回调驱动）"""
        self._running = True
    
    def stop(self):
        self._running = False


# ==================== TCP 模式适配器 ====================

class TCPAdapter(DeviceAdapter):
    """
    TCP Socket 设备适配器 - 通过 TCP 接入自定义设备
    
    协议约定：
    - 设备主动推送 JSON 数据，每行一条
    - 服务端监听指定端口，接收数据
    - 数据格式同 MQTT 适配器的 JSON 格式
    
    示例数据流：
    {"device_id":"radar_01","heart_rate":76,"spo2":98,...}\n
    {"device_id":"radar_01","heart_rate":78,"spo2":97,...}\n
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._server_socket = None
        self._client_socket = None
        self._recv_thread = None
    
    @property
    def mode_name(self) -> str:
        return "TCP 模式"
    
    def connect(self) -> bool:
        """启动 TCP 服务端监听"""
        import socket
        
        try:
            host = self.config.get("host", "0.0.0.0")
            port = self.config.get("port", 9527)
            
            self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._server_socket.settimeout(5.0)
            self._server_socket.bind((host, port))
            self._server_socket.listen(1)
            
            self._connection_status = "connecting"
            logger.info(f"TCP 服务端已启动，监听 {host}:{port}")
            return True
        except Exception as e:
            self._connection_status = "error"
            self._emit_error(f"TCP 服务端启动失败: {e}")
            return False
    
    def _accept_and_recv(self):
        """接受连接并接收数据"""
        while self._running:
            try:
                if self._client_socket is None:
                    self._connection_status = "connecting"
                    self._client_socket, addr = self._server_socket.accept()
                    self._client_socket.settimeout(10.0)
                    self._connection_status = "connected"
                    logger.info(f"TCP 设备已连接: {addr}")
                
                # 接收数据（按行分割）
                buffer = b""
                while self._running:
                    try:
                        chunk = self._client_socket.recv(4096)
                        if not chunk:
                            # 连接关闭
                            self._client_socket.close()
                            self._client_socket = None
                            self._connection_status = "disconnected"
                            break
                        
                        buffer += chunk
                        while b"\n" in buffer:
                            line, buffer = buffer.split(b"\n", 1)
                            line = line.strip()
                            if line:
                                self._process_line(line)
                    except socket.timeout:
                        continue
                    except ConnectionResetError:
                        self._client_socket.close()
                        self._client_socket = None
                        self._connection_status = "disconnected"
                        break
                        
            except socket.timeout:
                continue
            except Exception as e:
                self._emit_error(f"TCP 接收异常: {e}")
                time.sleep(1)
    
    def _process_line(self, line: bytes):
        """处理一行 JSON 数据"""
        try:
            payload = json.loads(line.decode('utf-8'))
            data = HealthData.from_dict(payload)
            self._emit_data(data)
        except json.JSONDecodeError:
            self._emit_error(f"TCP 数据解析失败: {line}")
        except Exception as e:
            self._emit_error(f"TCP 数据处理异常: {e}")
    
    def disconnect(self):
        self.stop()
        if self._client_socket:
            try:
                self._client_socket.close()
            except Exception:
                pass
            self._client_socket = None
        if self._server_socket:
            try:
                self._server_socket.close()
            except Exception:
                pass
            self._server_socket = None
        self._connection_status = "disconnected"
    
    def start(self):
        """开始接收 TCP 数据"""
        self._running = True
        self._recv_thread = threading.Thread(target=self._accept_and_recv, daemon=True)
        self._recv_thread.start()
    
    def stop(self):
        self._running = False
        if self._client_socket:
            try:
                self._client_socket.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
        if self._recv_thread:
            self._recv_thread.join(timeout=5)
            self._recv_thread = None


# ==================== 华为手环适配器 ====================

class HuaweiBandAdapter(MQTTAdapter):
    """
    华为手环专用适配器（基于 Gadgetbridge + MQTT）
    
    数据流：
    华为手环 → 蓝牙 → 手机 Gadgetbridge → MQTT Broker → 本适配器
    
    配置示例：
    {
        "mode": "huawei_band",
        "mqtt": {
            "broker": "localhost",
            "port": 1883,
            "topic_prefix": "gadgetbridge",
            "device_id": "huawei_band_001"
        },
        "user_profile": {
            "age": 65,
            "gender": "male",
            "resting_hr": 70
        }
    }
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.user_profile = config.get("user_profile", {})
        self.age = self.user_profile.get("age", 65)
        self.gender = self.user_profile.get("gender", "male")
        self.resting_hr = self.user_profile.get("resting_hr", 70)
        self._hr_history = []
        self._max_hr_history = 60  # 保留最近 60 次心率数据
    
    @property
    def mode_name(self) -> str:
        return "华为手环"
    
    def _on_message(self, client, userdata, msg):
        """处理 Gadgetbridge 推送的数据"""
        try:
            payload = json.loads(msg.payload.decode('utf-8'))
            
            # Gadgetbridge 数据格式解析
            heart_rate = self._parse_heart_rate(payload)
            spo2 = self._parse_spo2(payload)
            steps = self._parse_steps(payload)
            battery = self._parse_battery(payload)
            
            # 使用心率估算血压
            if heart_rate > 0:
                self._hr_history.append(heart_rate)
                if len(self._hr_history) > self._max_hr_history:
                    self._hr_history.pop(0)
                
                bp_s, bp_d = BloodPressureEstimator.estimate_from_history(
                    self._hr_history, self.age, self.gender
                )
            else:
                bp_s, bp_d = 120.0, 80.0
            
            # 构造标准 HealthData
            data = HealthData(
                heart_rate=heart_rate if heart_rate > 0 else 76.0,
                spo2=spo2 if spo2 > 0 else 98.0,
                bp_systolic=bp_s,
                bp_diastolic=bp_d,
                steps=steps,
                temperature=36.5,  # 华为手环通常不提供体温
                battery=battery if battery > 0 else 85.0,
                device_id=self._device_id or "huawei_band",
            )
            
            self._emit_data(data)
        except Exception as e:
            self._emit_error(f"华为手环数据解析失败：{e}")
    
    def _parse_heart_rate(self, payload: Dict) -> float:
        """从 Gadgetbridge 数据解析心率"""
        # Gadgetbridge 心率数据格式
        if "heart_rate" in payload:
            return float(payload["heart_rate"])
        if "hr" in payload:
            return float(payload["hr"])
        if "bpm" in payload:
            return float(payload["bpm"])
        return 0.0
    
    def _parse_spo2(self, payload: Dict) -> float:
        """从 Gadgetbridge 数据解析血氧"""
        if "spo2" in payload:
            return float(payload["spo2"])
        if "oxygen" in payload:
            return float(payload["oxygen"])
        return 0.0
    
    def _parse_steps(self, payload: Dict) -> int:
        """从 Gadgetbridge 数据解析步数"""
        if "steps" in payload:
            return int(payload["steps"])
        return 0
    
    def _parse_battery(self, payload: Dict) -> float:
        """从 Gadgetbridge 数据解析电量"""
        if "battery" in payload:
            return float(payload["battery"])
        if "level" in payload:
            return float(payload["level"])
        return 0.0


# ==================== 工厂函数 ====================

def create_adapter(config: Dict[str, Any]) -> DeviceAdapter:
    """
    根据配置创建对应的设备适配器
    
    Args:
        config: 设备配置字典，必须包含 "mode" 字段
            - "simulator": 模拟模式
            - "mqtt": MQTT 模式
            - "tcp": TCP 模式
    
    Returns:
        DeviceAdapter 实例
    """
    mode = config.get("mode", "simulator")
    
    if mode == "simulator":
        return SimulatorAdapter(config)
    elif mode == "mqtt":
        return MQTTAdapter(config)
    elif mode == "tcp":
        return TCPAdapter(config)
    elif mode == "huawei_band":
        return HuaweiBandAdapter(config)
    else:
        raise ValueError(f"未知的设备模式: {mode}，支持: simulator / mqtt / tcp / huawei_band")


# ==================== 配置管理 ====================

DEFAULT_CONFIG = {
    "mode": "simulator",
    "update_interval_ms": 2000,
    "simulator": {},
    "mqtt": {
        "broker": "localhost",
        "port": 1883,
        "username": "",
        "password": "",
        "client_id": "health_monitor",
        "topic_prefix": "health",
        "device_id": "",
    },
    "tcp": {
        "host": "0.0.0.0",
        "port": 9527,
    },
}

CONFIG_FILE_PATH = os.path.join(os.path.dirname(__file__), "device_config.json")


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """加载设备配置（优先从文件加载，不存在则使用默认配置）"""
    config_path = path or CONFIG_FILE_PATH
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"已加载设备配置: {config_path}")
            return config
        except Exception as e:
            logger.warning(f"配置文件加载失败，使用默认配置: {e}")
    
    return dict(DEFAULT_CONFIG)


def save_config(config: Dict[str, Any], path: Optional[str] = None):
    """保存设备配置到文件"""
    config_path = path or CONFIG_FILE_PATH
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    logger.info(f"设备配置已保存: {config_path}")


def create_default_config(path: Optional[str] = None):
    """创建默认配置文件（如果不存在）"""
    config_path = path or CONFIG_FILE_PATH
    if not os.path.exists(config_path):
        save_config(DEFAULT_CONFIG, config_path)
        logger.info(f"已创建默认配置文件: {config_path}")
    return config_path
