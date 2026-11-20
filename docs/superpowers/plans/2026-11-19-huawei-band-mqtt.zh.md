# 华为手环 MQTT 接入实施计划

> **给智能代理的说明：** 必须使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 按任务逐项实施本计划。步骤使用复选框（`- [ ]`）语法跟踪进度。

**目标：** 通过 Gadgetbridge 导出的 SQLite 数据库和 MQTT 接入华为手环 9 健康数据，并确保真实读数不会被模拟器数值替换。

**架构：** 独立桥接器从 `Gadgetbridge.db` 读取最新支持的指标，并发布标准化 JSON。`HuaweiBandAdapter` 按配置的主题订阅消息，将其转换为 `HealthData`，并从 MQTT 线程发出数据。`HealthDashboardDialog` 通过 Qt signal 转发回调，只渲染实际接收到的数值。

**技术栈：** Python 3.11+、PyQt6 signal、`paho-mqtt`、SQLite3、pytest。

---

### 任务 1：华为 MQTT 适配器合同

**文件：**
- 修改：`deidao/device_adapter.py`
- 测试：`test/test_huawei_band_adapter.py`
- 修改：`requirements.txt`
- 修改：`deidao/device_config.json`

- [x] 编写失败测试，证明 `huawei_band.mqtt` 控制 Broker 和主题配置、不支持的值保持缺失、血压估算必须显式启用、空消息不会发出伪造数据。
- [x] 运行 `.\.venv\Scripts\python.exe -m pytest test/test_huawei_band_adapter.py -q`，确认新测试先失败。
- [x] 在调用 `MQTTAdapter.__init__` 前规范化嵌套的华为配置；推导 `gadgetbridge/{device_id}/health`；拒绝空 payload；解析可选指标；只在配置启用时估算血压。
- [x] 添加 `paho-mqtt>=1.6,<3`，更新默认配置，并确认所有适配器测试通过。

### 任务 2：健康大屏真实数据路径

**文件：**
- 修改：`deidao/dialogs.py`
- 测试：`test/test_huawei_band_adapter.py`

- [x] 编写失败测试，覆盖 `NaN` 格式化为 `--`，以及应用一条真实样本时不调用模拟器随机更新。
- [x] 在对话框类中添加设备数据/设备错误 Qt signal，并在构造函数中连接它们。
- [x] 将渲染与模拟器更新拆开：MQTT 数据经 signal 入队并调用只负责渲染的方法；手动刷新仍使用模拟器模式。
- [x] 运行 `.\.venv\Scripts\python.exe -m pytest test/test_huawei_band_adapter.py -q`，确认测试通过。

### 任务 3：Gadgetbridge SQLite 桥接器

**文件：**
- 新建：`tools/gadgetbridge_mqtt_bridge.py`
- 测试：`test/test_gadgetbridge_mqtt_bridge.py`

- [x] 基于内存 SQLite 数据库编写失败测试，库中包含官方华为活动表结构以及设备电量表结构。
- [x] 实现只读最新样本提取、`NaN` 过滤、UTC ISO 时间戳、mtime 轮询、MQTT 保留消息发布和 `--once` 模式。
- [x] 运行 `.\.venv\Scripts\python.exe -m pytest test/test_gadgetbridge_mqtt_bridge.py -q`，确认测试通过。

### 任务 4：配置与验证

**文件：**
- 修改：`md/HUAWEI_BAND_GUIDE.md`
- 修改：`README.md`

- [x] 将“Gadgetbridge 内置 MQTT”的错误描述替换为数据库桥接架构和手机导出配置说明。
- [x] 运行 `.\.venv\Scripts\python.exe -m pytest test/test_device_adapter.py test/test_huawei_band_adapter.py test/test_gadgetbridge_mqtt_bridge.py -q`。
- [x] 运行 `.\.venv\Scripts\python.exe tools/gadgetbridge_mqtt_bridge.py --help`，检查命令输出。
