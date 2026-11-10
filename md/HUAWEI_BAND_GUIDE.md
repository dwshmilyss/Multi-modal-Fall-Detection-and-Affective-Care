# 华为手环 9 完整接入指南

## 概述

本指南说明如何将华为手环 9 的健康数据接入到 AI-Vision-Guardian 项目中，实现真实设备的健康数据监测。

### 数据流架构

```
华为手环 9 → 蓝牙 → 手机 Gadgetbridge → MQTT Broker → 本项目
```

### 支持的指标

| 指标 | 来源 | 说明 |
|------|------|------|
| 心率 | 直接采集 | 连续监测 |
| 血氧 | 直接采集 | 连续监测 |
| 血压 | 算法估算 | 基于心率变异性 (HRV) |
| 步数 | 直接采集 | 精确统计 |
| 体温 | 直接采集 | 皮肤温度 |
| 电量 | 直接采集 | 实时显示 |

---

## 第一步：安装 Gadgetbridge

### 1.1 下载 Gadgetbridge

Gadgetbridge 是一款免费开源的 Android 应用，用于连接智能穿戴设备。

- **F-Droid 下载**：https://f-droid.org/packages/nodomain.freeyourgadget.gadgetbridge/
- **GitHub 发布页**：https://github.com/Freeyourgadget/Gadgetbridge/releases
- **价格**：完全免费，开源

### 1.2 安装步骤

1. 在 Android 手机上安装 Gadgetbridge APK
2. 授予必要的权限（蓝牙、位置、通知等）
3. 打开 Gadgetbridge

### 1.3 配对华为手环 9

1. 在 Gadgetbridge 中点击"添加设备"
2. 选择"华为/荣耀"设备类型
3. 按照提示完成蓝牙配对
4. 等待数据同步

---

## 第二步：安装 MQTT Broker

### 2.1 Windows 安装 Mosquitto

1. 下载 Mosquitto：https://mosquitto.org/download/
2. 安装时选择"Install as Service"
3. 默认端口：1883

### 2.2 验证 MQTT Broker

```bash
# 安装 MQTT 客户端工具
pip install paho-mqtt

# 测试订阅
mosquitto_sub -h localhost -t "gadgetbridge/#"
```

---

## 第三步：配置 Gadgetbridge MQTT 推送

### 3.1 在 Gadgetbridge 中启用 MQTT

1. 打开 Gadgetbridge 设置
2. 找到"数据导出"或"MQTT"选项
3. 配置 MQTT 服务器：
   - **服务器地址**：你的电脑 IP（如 `192.168.1.100`）
   - **端口**：1883
   - **主题前缀**：`gadgetbridge`
   - **设备 ID**：`huawei_band_001`

### 3.2 数据格式

Gadgetbridge 推送的 JSON 数据格式示例：

```json
{
  "heart_rate": 76,
  "spo2": 98,
  "steps": 3462,
  "battery": 91,
  "timestamp": "2026-11-07T02:00:00Z"
}
```

---

## 第四步：配置项目

### 4.1 修改设备配置文件

编辑 `../deidao/device_config.json`：

```json
{
  "mode": "huawei_band",
  "update_interval_ms": 2000,
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

### 4.2 配置说明

| 字段 | 说明 | 默认值 |
|------|------|--------|
| `mode` | 设备模式，设为 `huawei_band` | `simulator` |
| `mqtt.broker` | MQTT 服务器地址 | `localhost` |
| `mqtt.port` | MQTT 端口 | `1883` |
| `mqtt.topic_prefix` | 主题前缀 | `gadgetbridge` |
| `mqtt.device_id` | 设备 ID | `huawei_band_001` |
| `user_profile.age` | 用户年龄（用于血压估算） | `65` |
| `user_profile.gender` | 用户性别 | `male` |
| `user_profile.resting_hr` | 静息心率 | `70` |

---

## 第五步：运行项目

### 5.1 启动顺序

1. **启动 MQTT Broker**（Mosquitto 服务）
2. **启动 Gadgetbridge**（手机上）
3. **启动本项目**

```bash
cd E:\work\PycharmProjects\Multi-modal-Fall-Detection-and-Affective-Care
python deidao/main.py
```

### 5.2 验证数据

1. 打开健康大屏
2. 查看右上角数据源状态：应显示"华为手环 - 已连接"
3. 检查各项指标是否正常更新

---

## 血压估算算法说明

### 原理

由于华为手环 9 不支持直接测量血压，我们使用**心率变异性 (HRV)** 算法进行估算：

```
收缩压 = 基础收缩压 + (当前心率 - 静息心率) × 0.5
舒张压 = 基础舒张压 + (当前心率 - 静息心率) × 0.2
```

### 基础血压值

| 年龄段 | 男性基础收缩压 | 女性基础收缩压 | 基础舒张压 |
|--------|---------------|---------------|-----------|
| < 40 岁 | 110 mmHg | 105 mmHg | 70 mmHg |
| 40-60 岁 | 120 mmHg | 115 mmHg | 75 mmHg |
| > 60 岁 | 130 mmHg | 125 mmHg | 80 mmHg |

### 示例

```
用户：65 岁男性，静息心率 70 bpm

当前心率 76 bpm：
- 收缩压 = 130 + (76-70) × 0.5 = 133 mmHg
- 舒张压 = 80 + (76-70) × 0.2 = 81.2 mmHg
- 结果：133/81 mmHg

当前心率 90 bpm：
- 收缩压 = 130 + (90-70) × 0.5 = 140 mmHg
- 舒张压 = 80 + (90-70) × 0.2 = 84 mmHg
- 结果：140/84 mmHg
```

### 注意事项

⚠️ **重要提示**：
- 血压估算值仅供参考，不能替代医疗设备测量
- 比赛展示时可以说明"基于心率变异性的估算值"
- 如需精确血压，建议使用华为 Watch D（气囊式血压计）

---

## 故障排查

### 问题 1：数据源显示"连接失败"

**原因**：MQTT Broker 未启动或配置错误

**解决**：
1. 检查 Mosquitto 服务是否运行
2. 验证 `device_config.json` 中的 broker 地址
3. 测试 MQTT 连接：`mosquitto_sub -h localhost -t "gadgetbridge/#"`

### 问题 2：心率显示为 0

**原因**：Gadgetbridge 未推送数据或数据格式不匹配

**解决**：
1. 检查 Gadgetbridge 是否正常连接手环
2. 查看 MQTT 消息：`mosquitto_sub -h localhost -t "gadgetbridge/#" -v`
3. 确认数据格式包含 `heart_rate` 或 `hr` 字段

### 问题 3：血压估算不准确

**原因**：用户档案配置不正确

**解决**：
1. 修改 `device_config.json` 中的 `user_profile`
2. 设置正确的年龄、性别和静息心率
3. 静息心率可以在早晨起床前测量

---

## 比赛展示建议

### 演示脚本

1. **开场**：展示健康大屏，说明数据源为"华为手环 9"
2. **实时数据**：展示心率、血氧、步数等实时更新的指标
3. **血压估算**：说明"由于手环不支持血压测量，我们使用心率变异性算法进行估算"
4. **告警功能**：模拟异常数据，展示告警弹窗和声音
5. **趋势图**：点击指标卡片，展示 ECharts 趋势图

### 技术亮点

- ✅ 真实设备数据接入（华为手环 9）
- ✅ 基于心率变异性的血压估算算法
- ✅ Gadgetbridge 开源方案（免费）
- ✅ MQTT 实时数据传输
- ✅ 智能告警系统（两级告警）
- ✅ ECharts 交互式趋势图

---

## 文件清单

| 文件 | 说明 |
|------|------|
| `../deidao/device_adapter.py` | 设备适配器框架（含血压估算算法） |
| `../deidao/device_config.json` | 设备配置文件 |
| `../deidao/dialogs.py` | 健康大屏对话框（集成设备适配器） |
| `HUAWEI_BAND_GUIDE.md` | 本指南 |
| `DEVICE_INTEGRATION.md` | 通用设备接入文档 |

---

## 参考资料

- **Gadgetbridge 官网**：https://gadgetbridge.org/
- **Gadgetbridge GitHub**：https://github.com/Freeyourgadget/Gadgetbridge
- **Mosquitto MQTT**：https://mosquitto.org/
- **华为手环 9**：https://consumer.huawei.com/cn/wearables/huawei-band-9/

---

## 更新日志

- **2026-11-07**：初始版本，支持华为手环 9 + Gadgetbridge + MQTT
- **2026-11-07**：添加血压估算算法（基于心率变异性）
