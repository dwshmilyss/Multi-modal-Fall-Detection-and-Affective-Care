# 设备接入文档

## 概述

本项目支持多种设备数据源，通过统一的适配器框架实现无缝切换。

### 支持的设备模式

| 模式 | 适配器类 | 说明 |
|------|---------|------|
| `simulator` | SimulatorAdapter | 模拟数据生成器（默认） |
| `mqtt` | MQTTAdapter | MQTT 协议接入真实设备 |
| `tcp` | TCPAdapter | TCP Socket 接入自定义设备 |
| `huawei_band` | HuaweiBandAdapter | 华为手环专用（Gadgetbridge + MQTT） |

---

## 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                     HealthDashboardDialog                    │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────┐  │
│  │              DeviceAdapter (接口)                     │  │
│  │  ┌─────────────┬─────────────┬────────────────────┐ │  │
│  │  │ Simulator   │   MQTT      │   TCP              │ │  │
│  │  │ Adapter     │   Adapter   │   Adapter          │ │  │
│  │  └─────────────┴─────────────┴────────────────────┘ │  │
│  │  ┌────────────────────────────────────────────────┐ │  │
│  │  │         HuaweiBandAdapter                      │ │  │
│  │  │    (继承 MQTTAdapter + 血压估算)                │ │  │
│  │  └────────────────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 配置文件

### device_config.json

```json
{
  "mode": "simulator",
  "update_interval_ms": 2000,
  
  "simulator": {
    "update_interval_ms": 2000
  },
  
  "mqtt": {
    "broker": "localhost",
    "port": 1883,
    "username": "",
    "password": "",
    "client_id": "health_monitor_client",
    "topic_prefix": "health",
    "device_id": "band_001"
  },
  
  "tcp": {
    "host": "0.0.0.0",
    "port": 9527
  },
  
  "huawei_band": {
    "mqtt": {
      "broker": "localhost",
      "port": 1883,
      "username": "",
      "password": "",
      "client_id": "huawei_band_client",
      "topic_prefix": "gadgetbridge",
      "device_id": "huawei_band_001"
    },
    "user_profile": {
      "age": 65,
      "gender": "male",
      "resting_hr": 70
    }
  }
}
```

---

## 数据模型

### HealthData

所有适配器输出统一的 `HealthData` 数据模型：

```python
@dataclass
class HealthData:
    heart_rate: float      # 心率 (bpm)
    spo2: float            # 血氧饱和度 (%)
    bp_systolic: float     # 收缩压 (mmHg)
    bp_diastolic: float    # 舒张压 (mmHg)
    steps: int             # 步数
    temperature: float     # 体温 (°C)
    battery: float         # 电量 (%)
    device_id: str         # 设备 ID
    timestamp: str         # 时间戳
```

---

## 适配器开发指南

### 基类接口

```python
class DeviceAdapter(ABC):
    @abstractmethod
    def connect(self) -> bool:
        """连接设备"""
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
        """设备模式名称"""
        pass
    
    def on_data(self, callback: Callable[[HealthData], None]):
        """注册数据回调"""
        pass
    
    def on_error(self, callback: Callable[[str], None]):
        """注册错误回调"""
        pass
```

### 自定义适配器示例

```python
class MyCustomAdapter(DeviceAdapter):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # 初始化自定义逻辑
    
    @property
    def mode_name(self) -> str:
        return "自定义设备"
    
    def connect(self) -> bool:
        # 实现连接逻辑
        return True
    
    def disconnect(self):
        # 实现断开逻辑
        pass
    
    def start(self):
        # 开始接收数据
        # 收到数据后调用 self._emit_data(health_data)
        pass
    
    def stop(self):
        # 停止接收数据
        pass
```

### 注册到工厂函数

在 `../deidao/device_adapter.py` 的 `create_adapter` 函数中添加：

```python
elif mode == "my_custom":
    return MyCustomAdapter(config)
```

---

## MQTT 数据格式

### 订阅主题

```
{topic_prefix}/{device_id}/health
```

示例：`health/band_001/health`

### 消息格式

```json
{
  "heart_rate": 76,
  "spo2": 98,
  "bp_systolic": 120,
  "bp_diastolic": 80,
  "steps": 3462,
  "temperature": 36.5,
  "battery": 91,
  "device_id": "band_001",
  "timestamp": "2026-11-07T02:00:00Z"
}
```

### 可选字段

- `heart_rate` / `hr` / `bpm`：心率（任一字段均可）
- `spo2` / `oxygen`：血氧（任一字段均可）
- `battery` / `level`：电量（任一字段均可）

---

## TCP 数据格式

### 连接信息

- **主机**：`0.0.0.0`（可配置）
- **端口**：`9527`（可配置）
- **协议**：设备主动推送 JSON 数据，每行一条

### 数据格式

每行一个 JSON 对象，格式同 MQTT 消息格式。

---

## 华为手环专用

### 数据流

```
华为手环 9 → 蓝牙 → 手机 Gadgetbridge → MQTT Broker → 本项目
```

### Gadgetbridge 配置

1. 下载 Gadgetbridge：https://f-droid.org/packages/nodomain.freeyourgadget.gadgetbridge/
2. 配对华为手环 9
3. 启用 MQTT 推送：
   - 服务器：电脑 IP
   - 端口：1883
   - 主题前缀：`gadgetbridge`

### 血压估算

华为手环 9 不支持直接测量血压，使用心率变异性 (HRV) 算法估算：

```python
# 收缩压 = 基础收缩压 + (心率 - 静息心率) × 0.5
# 舒张压 = 基础舒张压 + (心率 - 静息心率) × 0.2
```

详见 `HUAWEI_BAND_GUIDE.md`。

---

## 测试

### 运行测试脚本

```bash
python test_device_adapter.py
```

### 测试内容

- HealthData 数据模型
- 配置管理（加载/保存）
- 工厂函数
- 模拟器适配器
- 血压估算算法
- 华为手环适配器

---

## 故障排查

### 常见问题

1. **数据源显示"连接失败"**
   - 检查 MQTT Broker 是否运行
   - 验证配置文件中的 broker 地址

2. **心率显示为 0**
   - 检查 Gadgetbridge 是否正常连接
   - 查看 MQTT 消息格式是否正确

3. **血压估算不准确**
   - 检查 `user_profile` 配置（年龄、性别、静息心率）

---

## 参考资料

- **Gadgetbridge**：https://gadgetbridge.org/
- **Mosquitto MQTT**：https://mosquitto.org/
- **Paho MQTT Python**：https://pypi.org/project/paho-mqtt/

---

## 更新日志

- **2026-11-07**：初始版本，支持 simulator/mqtt/tcp 三种模式
- **2026-11-07**：添加华为手环适配器（huawei_band）和血压估算算法
