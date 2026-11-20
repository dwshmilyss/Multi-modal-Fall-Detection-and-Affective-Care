# 华为手环 9 接入与运行指南

本文描述当前代码的实际接入方式。Gadgetbridge 没有“手环数据直接 MQTT 推送”功能，因此数据链路是：

```text
华为手环 9
  → Android 手机 Gadgetbridge
  → Gadgetbridge 导出 SQLite 数据库
  → 数据库文件同步到 Windows
  → tools/gadgetbridge_mqtt_bridge.py 读取最新数据
  → Mosquitto MQTT Broker
  → deidao/device_adapter.py 订阅 MQTT
  → PyQt6 健康大屏显示
```

手机不连接 MQTT。MQTT 只存在于 Windows 本机：桥接器发布，本项目订阅。

## 1. 当前支持范围

| 指标 | 当前结果 | 说明 |
| --- | --- | --- |
| 心率 | 支持 | 从 `HUAWEI_ACTIVITY_SAMPLE.HEART_RATE` 读取最新有效值 |
| 血氧 | 支持 | 从 `HUAWEI_ACTIVITY_SAMPLE.SPO` 读取最新有效值 |
| 今日步数 | 支持 | 汇总当天本地 0 点后的 `STEPS` |
| 电量 | 有则显示 | 数据库中有电量表或电量属性时读取 |
| 体温 | 显示 `--` | 当前桥接器不承诺手环 9 体温 |
| 血压 | 显示 `--` | 手环 9 不支持血压测量；估算默认关闭 |

重要限制：

- 这不是实时原始传感器流，不能用手环 9 加速度计做实时跌倒检测。
- 数据新鲜度取决于 Gadgetbridge 同步和数据库导出频率，通常是分钟级，不是 2 秒级。
- 健康大屏显示“华为手环 - 已连接”只表示本项目连上了 MQTT Broker，不代表手机正在实时连接手环。
- 缺失字段显示 `--`，不会用演示数据冒充真实数据。

### 1.2 比赛演示兜底

`deidao/device_config.json` 中华为模式默认包含：

```json
"fallback_to_simulator": true
```

该配置的含义是：

1. 健康大屏仍会连接 MQTT 并订阅华为手环 topic。
2. 如果暂时没有真实手环消息，界面立即恢复原有 2 秒模拟数据刷新，保证比赛可以演示。
3. 顶部状态会明确显示“已连接，模拟演示等待真实数据”，不会把模拟数据说成真实手环数据。
4. 第一条真实 MQTT 数据到达后，模拟定时器立即停止，历史缓冲清空，界面切换为真实数据。
5. 真实数据到达后，手动刷新也不会再用模拟器随机值覆盖真实值。

如果现场需要严格等待真实数据、不允许演示模拟值，将该项改为：

```json
"fallback_to_simulator": false
```

关闭后，MQTT 连接成功但无真实消息时，界面保持等待状态，不会启动模拟刷新。

### 1.1 数据库说明与安装检查

当前手环接入链路使用的是 **SQLite 单文件数据库**，文件名通常为 `Gadgetbridge.db`。

| 问题 | 结论 |
| --- | --- |
| 数据库由谁生成？ | Android 手机上的 Gadgetbridge App 自动生成和维护 |
| Windows 需要安装 SQLite 服务吗？ | 不需要，SQLite 是单文件数据库，没有服务端 |
| 需要额外安装 Python SQLite 包吗？ | 不需要，Python 标准库自带 `sqlite3` 模块 |
| 桥接器会如何访问数据库？ | 只读方式打开，不修改手机导出的数据库 |
| 手环数据会写入项目 MySQL 吗？ | 不会，当前健康数据链路不使用 MySQL |
| 需要手动建表吗？ | 不需要，表结构由 Gadgetbridge 导出文件自带 |

本项目主程序原有的 MySQL 用于登录、跌倒告警历史等业务数据。华为手环健康数据接入只读取 `Gadgetbridge.db` 并通过 MQTT 传给健康大屏，不要把手环数据库配置到 MySQL 中。

在 Windows 上只需要检查 Python 自带的 SQLite 能力：

```powershell
.\.venv\Scripts\python.exe -c "import sqlite3; print(sqlite3.sqlite_version)"
```

能输出版本号即可，例如：

```text
3.42.0
```

不需要执行 `pip install sqlite3`。如果这条命令失败，说明 Python 环境不完整，应重新创建 Python 3.11 虚拟环境。

可选：如果你想在电脑上人工查看 `Gadgetbridge.db`，可以安装 DB Browser for SQLite，但它不是本方案运行依赖。桥接器只需要这个文件存在且包含 Gadgetbridge 导出的数据。

## 2. 准备 Windows 运行环境

项目运行基线是 Python 3.11。当前仓库自带 `.venv`，应为 Python 3.11.6。

```powershell
cd E:\work\PycharmProjects\Multi-modal-Fall-Detection-and-Affective-Care
.\.venv\Scripts\python.exe --version
```

预期输出 `Python 3.11.6`。如果 `.venv` 不存在或版本不是 3.11：

```powershell
py -3.11 -m venv .venv
```

安装依赖并运行接入测试：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest test/test_device_adapter.py test/test_huawei_band_adapter.py test/test_gadgetbridge_mqtt_bridge.py -q
```

预期结果为 `19 passed`。后续所有 Python 命令都显式使用 `.venv`，避免误用系统里的其他 Python 版本。

## 3. 配置 Android 手机和 Gadgetbridge

### 3.1 安装并配对手环

1. 在 Android 手机上安装 Gadgetbridge。优先使用 F-Droid 稳定版；如果稳定版不能识别手环 9，改用官方 nightly 版本。
2. 手环先从华为运动健康解绑。解绑可能清除表盘、通知和个性化设置。
3. 打开 Gadgetbridge，进入发现或添加设备流程，找到华为手环 9 并完成配对。
4. 授予蓝牙、位置、后台运行、自启动等必要权限。
5. 在 Android 系统设置中将 Gadgetbridge 的电池优化设为不受限制或允许后台运行。
6. 回到 Gadgetbridge，确认手环状态为已连接或已初始化。

### 3.2 开启健康数据同步

在 Gadgetbridge 的设备设置中开启：

1. 心率自动测量或连续心率。
2. 血氧自动测量。
3. 活动数据、睡眠数据自动同步。
4. 手动触发一次活动数据同步，确保数据库中已有样本。

不同 Gadgetbridge 版本的菜单名称可能略有差异，以功能含义为准。

### 3.3 导出 Gadgetbridge 数据库

在 Gadgetbridge 设置中找到自动化、数据导出或数据库导出相关选项：

1. 设置一个手机上可访问的导出目录，例如 `Download/Gadgetbridge`。
2. 手动执行一次数据库导出。
3. 确认目录中出现 SQLite 数据库文件。本文后续统一称它为 `Gadgetbridge.db`。
4. 如果实际文件名不同，将后文 `--db` 参数替换为实际路径即可。

优先使用 Gadgetbridge 自带导出功能，不要在应用写入内部数据库时直接抄取应用私有数据。

## 4. 把数据库同步到 Windows

创建固定目录：

```powershell
New-Item -ItemType Directory -Force -Path D:\Gadgetbridge
$DB = "D:\Gadgetbridge\Gadgetbridge.db"
```

### 方式 A：手动复制，适合首次验证和演示

1. 手机连接电脑。
2. 从手机导出目录复制 `Gadgetbridge.db` 到 `D:\Gadgetbridge\Gadgetbridge.db`。
3. 每次更新数据时，重复：Gadgetbridge 同步、导出数据库、覆盖复制 Windows 文件。

如果已开启 USB 调试，可以使用 `adb pull`。假设手机导出目录是 `/sdcard/Download/Gadgetbridge/`：

```powershell
adb devices
adb pull /sdcard/Download/Gadgetbridge/Gadgetbridge.db $DB
```

`/sdcard/Download/Gadgetbridge/Gadgetbridge.db` 是示例路径，请替换为你在 Gadgetbridge 中设置的导出路径。

### 方式 B：Syncthing 自动同步，适合长时间运行

1. Android 手机和 Windows 都安装 Syncthing。
2. 两端互相配对。
3. Android 端把 Gadgetbridge 导出目录同步到 Windows 的 `D:\Gadgetbridge`。
4. 确保手机端会按预期频率同步手环数据并更新导出的数据库文件。
5. Windows 桥接器会按 `--poll-seconds` 检查文件变化并重新发布。

Syncthing 只负责同步文件，不会触发 Gadgetbridge 从手环拉数据，也不会替 Gadgetbridge 导出数据库。

## 5. 安装并启动 Mosquitto

1. 在 Windows 安装 Mosquitto。默认安装目录通常为 `C:\Program Files\mosquitto\`。
2. 启动服务：

```powershell
net start mosquitto
```

3. 检查状态：

```powershell
Get-Service mosquitto
```

预期状态为 `Running`。

4. 可选：打开实时订阅窗口：

```powershell
& "C:\Program Files\mosquitto\mosquitto_sub.exe" -h 127.0.0.1 -t "gadgetbridge/#" -v
```

该命令会一直等待消息，保留窗口即可。

## 6. 配置本项目

编辑 `deidao/device_config.json`，确认关键字段：

```json
{
  "mode": "huawei_band",
  "huawei_band": {
    "mqtt": {
      "broker": "localhost",
      "port": 1883,
      "username": "",
      "password": "",
      "client_id": "huawei_band_client",
      "topic_prefix": "gadgetbridge",
      "device_id": "huawei_band_9"
    },
    "user_profile": {
      "age": 65,
      "gender": "male",
      "resting_hr": 70
    },
    "estimate_blood_pressure": false,
    "fallback_to_simulator": true
  }
}
```

配置要求：

- `mode` 必须是 `huawei_band`。
- `broker` 保持 `localhost`，因为桥接器和主程序都在这台 Windows 电脑上运行。
- `topic_prefix` 保持 `gadgetbridge`。
- `device_id` 保持 `huawei_band_9`。
- `estimate_blood_pressure` 保持 `false`。如改为 `true`，血压也只是基于心率的粗略估算，不能作为医疗依据。
- `fallback_to_simulator` 保持 `true`，比赛或演示时无需切换回模拟模式。

如果 Mosquitto 配置了账号密码，`username` 和 `password` 必须与 Broker 一致，并且桥接器启动时也要传入相同凭据。

## 7. 首次验证数据库可读

先不连接 Broker，只检查数据库能否读出手环数据：

```powershell
$DB = "D:\Gadgetbridge\Gadgetbridge.db"
.\.venv\Scripts\python.exe tools\gadgetbridge_mqtt_bridge.py --db $DB --once --dry-run
```

成功时会输出类似：

```text
gadgetbridge/huawei_band_9/health retain=True {"device_id":"huawei_band_9","timestamp":"2026-11-20T04:30:00+00:00","heart_rate":82,"spo2":97,"steps":4100,"battery":76}
```

验收标准：

- 出现 `gadgetbridge/huawei_band_9/health`。
- JSON 中至少出现 `heart_rate`、`spo2`、`steps` 中的一个。
- `timestamp` 对应最新有效样本的 UTC 时间。

如果提示数据库不存在，检查 `$DB` 路径。如果提示缺少 `HUAWEI_ACTIVITY_SAMPLE`，回到手机端完成配对和活动数据同步后重新导出。如果输出“没有可发布的手环数据”，说明结构识别成功但缺少当天有效样本。

## 8. 完整运行步骤

建议使用三个 PowerShell 窗口，按顺序启动。

### 窗口 1：启动 MQTT 并观察消息

```powershell
cd E:\work\PycharmProjects\Multi-modal-Fall-Detection-and-Affective-Care
net start mosquitto
& "C:\Program Files\mosquitto\mosquitto_sub.exe" -h 127.0.0.1 -t "gadgetbridge/#" -v
```

保留该窗口，稍后应能看到桥接器发布的 JSON。

### 窗口 2：启动数据库桥接器

```powershell
cd E:\work\PycharmProjects\Multi-modal-Fall-Detection-and-Affective-Care
$DB = "D:\Gadgetbridge\Gadgetbridge.db"
.\.venv\Scripts\python.exe tools\gadgetbridge_mqtt_bridge.py `
  --db $DB `
  --broker 127.0.0.1 `
  --port 1883 `
  --device-query "huawei band 9" `
  --device-id huawei_band_9 `
  --topic-prefix gadgetbridge `
  --poll-seconds 5
```

桥接器启动后会立即读取一次数据库并发布保留消息；之后每 5 秒检查文件是否变化，只有 JSON 内容变化时才重新发布。

窗口 1 应看到类似：

```text
gadgetbridge/huawei_band_9/health {"device_id":"huawei_band_9","timestamp":"2026-11-20T04:30:00+00:00","heart_rate":82,"spo2":97,"steps":4100}
```

### 窗口 3：启动主程序

```powershell
cd E:\work\PycharmProjects\Multi-modal-Fall-Detection-and-Affective-Care
.\.venv\Scripts\python.exe deidao\main.py
```

1. 登录系统。
2. 打开健康大屏。
3. 顶部应显示“数据源：华为手环 - 已连接”。
4. 心率、血氧、步数应与 MQTT JSON 一致。
5. 体温和血压显示 `--` 是当前默认预期行为。

如果 Broker 不可达或连接失败，系统会回退到模拟模式。此时不能把界面数值当作真机数据。

### 让数据更新

1. 手机上的 Gadgetbridge 连接手环并同步活动数据。
2. Gadgetbridge 导出数据库。
3. Syncthing 或手动复制覆盖 `D:\Gadgetbridge\Gadgetbridge.db`。
4. 桥接器检测到文件变化后重新发布 MQTT。
5. 健康大屏收到新的 `HealthData` 并更新界面。

## 9. 一分钟演示流程

适合比赛或现场验收：

1. 启动 Mosquitto。
2. 启动 `mosquitto_sub` 观察消息。
3. 手机 Gadgetbridge 手动同步手环数据。
4. 手机 Gadgetbridge 导出数据库。
5. 将新的 `Gadgetbridge.db` 复制到 `D:\Gadgetbridge\Gadgetbridge.db`。
6. 发布一次：

```powershell
.\.venv\Scripts\python.exe tools\gadgetbridge_mqtt_bridge.py `
  --db D:\Gadgetbridge\Gadgetbridge.db `
  --once `
  --broker 127.0.0.1 `
  --device-id huawei_band_9
```

7. 启动或刷新健康大屏。
8. 对比 MQTT JSON 与大屏数值。

## 10. 故障排查

### Python 或依赖错误

```powershell
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 数据库文件不存在

```powershell
Test-Path D:\Gadgetbridge\Gadgetbridge.db
```

应返回 `True`。如果为 `False`，说明手机导出或复制步骤没有完成。

### 提示缺少 `HUAWEI_ACTIVITY_SAMPLE`

说明当前数据库不是已同步华为手环数据的 Gadgetbridge 数据库。执行：手机端手动同步活动数据、重新导出数据库、覆盖 Windows 数据库文件。

### dry-run 有数据，MQTT 订阅没有数据

1. 确认 Mosquitto 服务为 `Running`。
2. 确认桥接器没有连接错误。
3. 确认完整 topic 为 `gadgetbridge/huawei_band_9/health`。
4. 确认桥接器参数包含 `--topic-prefix gadgetbridge` 和 `--device-id huawei_band_9`。

### MQTT 有数据，健康大屏没有数据

1. 确认 `deidao/device_config.json` 中 `mode` 是 `huawei_band`。
2. 确认配置中的 `topic_prefix` 和 `device_id` 也是 `gadgetbridge / huawei_band_9`。
3. 确认主程序没有回退到模拟模式。
4. 重启主程序，让适配器重新连接 Broker 并订阅 topic。

### 大屏一直显示旧值

检查 Windows 数据库文件是否真的更新：

```powershell
Get-Item D:\Gadgetbridge\Gadgetbridge.db | Select-Object LastWriteTime,Length
```

如果时间不变，问题在手机同步、导出或文件同步，不在 MQTT。

### 电量显示 `--`

这是允许出现的结果，表示当前导出数据库中没有可读取的电量数据，不要用固定值替代。

### 心率或血氧显示 `--`

1. 在 Gadgetbridge 中手动测量一次心率或血氧。
2. 同步活动数据。
3. 重新导出数据库并覆盖 Windows 文件。
4. 观察桥接器是否重新发布。

## 11. 工程验证命令

```powershell
.\.venv\Scripts\python.exe -m pytest test/test_device_adapter.py test/test_huawei_band_adapter.py test/test_gadgetbridge_mqtt_bridge.py -q
.\.venv\Scripts\python.exe -m py_compile deidao\device_adapter.py deidao\dialogs.py tools\gadgetbridge_mqtt_bridge.py
.\.venv\Scripts\python.exe tools\gadgetbridge_mqtt_bridge.py --help
```
