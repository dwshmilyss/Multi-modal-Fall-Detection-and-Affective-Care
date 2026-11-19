# AI-Vision-Guardian: Multi-modal Fall Detection and Affective Care

> **AI-Vision-Guardian: Multi-modal Fall Detection and Affective Care** — a desktop elderly-care system combining **YOLOv8 + MediaPipe Pose + PyQt6 + MySQL** with **MediaPipe + ViT facial emotion recognition**, **Vosk offline speech recognition**, **Kimi LLM** and **pyttsx3 / SAPI5 Chinese TTS**.
>
> 适用于智慧养老、智能安防、毕业设计等场景；检测结果不应作为医疗诊断或唯一安全决策依据。

![Platform](https://img.shields.io/badge/platform-Windows%2010%2F11-0078d4)
![Python](https://img.shields.io/badge/python-3.8%E2%80%933.11-3776ab)
![License](https://img.shields.io/badge/license-MIT-green)
![Stack](https://img.shields.io/badge/stack-PyQt6%20%7C%20Ultralytics%20%7C%20MediaPipe%20%7C%20MySQL-orange)

---

## ✨ 功能特性

### 🛡️ 核心检测（`deidao/`）
| 能力 | 说明 |
| --- | --- |
| YOLOv8 实时人员检测 | COCO `person` 类，画定位框 |
| MediaPipe Pose 关键点 | 双肩、双髋中点计算**躯干倾角** |
| **时序状态机跌倒检测** | 按人员 ID 维护真实时间序列；动态特征 + 低姿态终末确认，必须达到加权共识才报警 |
| **坐姿/躺卧区分** | 稳定椅子高度髋位识别坐姿；缓慢躺下识别为平躺而不立即报警，快速跌倒才确认告警 |
| **尺度与 FPS 自适应** | 速度按“站立身高/秒”和真实时间计算，降低分辨率、距离、帧率差异影响 |
| **多人状态隔离** | 轻量 IoU 跟踪器为每个人员维持独立历史，避免遮挡或多人时串状态 |
| **告警保持** | 确认跌倒后 UI 告警保持 3 秒，避免只闪一帧；姿态缺失时候选分自动衰减 |
| 前置缓冲区录像 | 跌倒告警自动拼接 N 秒前置视频，存盘至 `result/` |
| MySQL 历史回放 | `alarm_logs` 表可追溯，含家属处理意见 |
| 飞书机器人推送 | 检测到跌倒即向飞书群推送交互卡片（时间/录像存档/系统状态） |
| 双角色系统 | 管理员运维端（`root`）/ 家属亲情看护端（`family`） |
| 健康指标大屏 | 心率/血氧/血压/步数/体温/电量，每 2 秒刷新；点击卡片看 ECharts 趋势图，可导出 CSV 报告 |
| **智能健康告警** | 警告/危险两级告警 + 弹窗 + 告警音，5 秒冷却防轰炸，可静音，含记录面板与统计 |
| **华为手环 9 接入** | Gadgetbridge + MQTT 真实设备数据流，心率变异性（HRV）血压估算；默认为模拟器模式 |
| 中文字体自适应 | `msyh.ttc` / `simhei.ttf` 自动回退 |

> 📄 详细文档（`md/` 目录）：[跌倒检测算法优化](md/FALL_DETECTION_OPTIMIZATION.md) · [智能告警系统](md/ALARM_SYSTEM_README.md) · [健康大屏](md/HEALTH_DASHBOARD_README.md) · [华为手环 9 接入指南](md/HUAWEI_BAND_GUIDE.md) · [飞书机器人配置指南](md/飞书机器人配置指南.md)

### 💬 语音 & AI（`deidao/dialogs.py`）
| 能力 | 说明 |
| --- | --- |
| **情绪识别弹窗** | MediaPipe 检测人脸 + ViT ONNX 实时识别 `高兴/悲伤/愤怒/惊讶/中性/恐惧/厌恶`，可据此调出 AI 诊断建议 |
| **AI 智能问答** | 打字 + 麦克风（Vosk 离线 / Google 在线）双输入 |
| **Kimi 大模型** | 通过环境变量或 `config_kimi.txt` 注入 API Key，自动回退到本地规则模板 |
| **pyttsx3 中文 TTS** | 一次性 init → 用 → 销毁（避免 SAPI5 engine 复用哑火）；逐句可中断，停止按钮 100~300ms 收声 |

### 🧰 工程工具
| 能力 | 说明 |
| --- | --- |
| `test/test_alarm_system.py` | 告警系统测试：级别区分 / 冷却机制 / 静音 / 统计 |
| `test/test_device_adapter.py` | 设备适配器测试（模拟器 / 华为手环 MQTT） |
| `test/test_health_dashboard.py` | 健康大屏独立测试 |
| `test/test_emotion_pipeline.py` | 情绪识别管线测试 |
| 模块化拆分 | `main.py / ui_layout.py / dialogs.py / detector.py / database.py` 单一职责 |

---

## 🖼️ 项目结构

```text
AI-Vision-Guardian/
├── deidao/                       # 跌倒检测主系统
│   ├── main.py                   # PyQt6 主程序入口
│   ├── ui_layout.py              # One Dark 风格主界面
│   ├── dialogs.py                # 6 类弹窗 + TTS / 语音 / AI 多线程
│   ├── detector.py               # YOLOv8+MediaPipe 检测线程（多特征融合算法）
│   └── database.py               # MySQL 初始化
├── qingxu/                       # 情绪识别子模块
│   ├── emotion_pipeline.py       # MediaPipe 人脸检测 + ViT 七分类实时管线
│   ├── pro_emotion/models/onnx/model.onnx # ViT 情绪权重
│   └── app.py                    # YOLOv11 face emotion 对照 demo
├── md/                           # 项目文档
│   ├── FALL_DETECTION_OPTIMIZATION.md  # 跌倒检测算法优化（v1.0→v6.0 演进）
│   ├── ALARM_SYSTEM_README.md    # 智能告警系统说明
│   ├── HEALTH_DASHBOARD_README.md # 健康大屏说明
│   ├── HUAWEI_BAND_GUIDE.md      # 华为手环 9 接入指南
│   └── 飞书机器人配置指南.md      # 飞书告警推送配置
├── model/
│   └── vosk-model-small-cn-0.22/ # Vosk 中文模型（首次使用需下载，见下文）
├── test/                         # 功能测试（告警 / 设备适配 / 大屏 / 情绪管线）
├── video/                        # 示例视频（不入库）
├── result/                       # 告警片段（运行时生成）
├── yolov8n.pt                    # YOLOv8 Nano 权重（需放置）
├── alarm.mp3                     # 告警音频（可替换）
├── requirements.txt              # 一键依赖安装
├── .env.example                  # 环境变量样例（KIMI_API_KEY 等）
├── LICENSE                       # MIT
└── README.md                     # 本说明
```

---

## 🧩 系统架构

```
┌──────────────────────────────────────────────────────────────────┐
│  PyQt6 主窗口 (deidao/main.py)                                  │
│                                                                  │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐             │
│  │ 检测线程     │  │ 情绪识别弹窗  │  │ AI 智能问答   │             │
│  │ (detector)  │  │ (ViT+Face)   │  │ (Vosk+SR)   │             │
│  │ YOLO+Pose   │  │              │  │ + Kimi LLM  │             │
│  └─────┬───────┘  └──────┬───────┘  └──────┬──────┘             │
│        │ 告警事件        │ 情绪           │ AI 回复               │
│        ↓                ↓                ↓                      │
│  ┌──────────────────────────────────────────────────┐           │
│  │ result/xxx.mp4 + MySQL alarm_logs + 飞书 webhook │           │
│  └──────────────────────────────────────────────────┘           │
│                                                                  │
│  TTS 引擎：pyttsx3 + SAPI5（中文音色）                           │
│   - 每个 worker init → 用 → 销毁（一次性生命周期）                │
│   - 句间可中断，100~300ms 收声                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

## ⚙️ 环境要求

| 项 | 要求 |
| --- | --- |
| OS | Windows 10 / 11（SAPI5 仅 Windows，原生支持中文 TTS） |
| Python | 3.8 ~ 3.11 |
| MySQL | 5.7+ / 8.x，本地 `localhost:3306`（若无需数据库可禁用） |
| 硬件 | 摄像头 / 麦克风 / 扬声器（可选） |
| GPU | 可选；CPU 也能跑（建议 RTX 1650 以上） |

---

## 🚀 快速开始

### 1. 克隆与虚拟环境

```bash
git clone https://github.com/<your-username>/AI-Vision-Guardian.git
cd AI-Vision-Guardian

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux（如跨平台调试）
# source .venv/bin/activate

pip install -r requirements.txt
```

### 2. 下载大文件（不进 Git）

| 文件 | 来源 | 放置位置 |
| --- | --- | --- |
| `yolov8n.pt` | [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) | 项目根目录 |
| `vosk-model-small-cn-0.22` | [Vosk Models](https://alphacephei.com/vosk/models) | `model/vosk-model-small-cn-0.22/` |
| `model.onnx`（情绪识别） | `trpakov/vit-face-expression` 导出 | `qingxu/pro_emotion/models/onnx/model.onnx` |
| `msyh.ttc` 或 `simhei.ttf` | Windows 系统字体 / 自行放置 | 项目根目录 |

> Ultralytics 缺权重时通常会自动下载；Vosk 模型需手动放置。当前工作区的 ViT 权重已在上表位置；新克隆仓库需按上表恢复该文件。

### 3. 配置环境变量

```bash
copy .env.example .env
# 编辑 .env，按需填入（均可留空，自动回退默认行为）：
#   KIMI_API_KEY=sk-...        # Kimi 大模型（无 Key 走本地规则模板）
#   FEISHU_WEBHOOK=https://... # 飞书跌倒告警推送（配置步骤见「健康监测子系统」）
```

也可以把 Kimi Key 放进 `deidao/config_kimi.txt`（一行一 key，无空格）。

### 4. （可选）准备 MySQL

启动本地 MySQL，首次运行会自动创建 `fall_detector_db` 数据库与默认账号：

| 用户名 | 密码 | 角色 |
| --- | --- | --- |
| `root` | `123456` | 管理员 |
| `family` | `123456` | 家属 |

> 默认密码**仅供本地演示**，正式部署请改用 `os.environ` 注入或加密配置。

### 5. 启动应用

```bash
# 跌倒检测主系统
cd deidao && python main.py

# 情绪识别独立 demo
cd qingxu && python app.py

# 功能测试（告警 / 设备适配 / 健康大屏 / 情绪管线）
python -m pytest test/
```

---

## 🧪 使用说明

### 管理员运维端

1. 用 `root` 登录
2. 左侧选**摄像头**或**导入视频**
3. 点**启动检测**，观察右侧实时姿态、告警卡片与日志
4. **系统设置**：调节躯干倾角阈值、声音告警、录制前置缓冲、飞书 Webhook
5. **历史片段**：在程序内查看 / 打开 `result/` 目录

### 家属亲情看护端

1. 用 `family` 登录
2. 实时监测可用，但部分参数被锁定
3. **飞书音视频通话**：一键调起飞书客户端
4. **健康大屏**：心率/血氧/血压/步数/体温/电量 + 趋势图 + 告警记录（详见「健康监测子系统」）
5. **历史片段**：可填写处理备注并标记"已处理"

### 情绪识别弹窗（任意角色可用）

1. 启动后调用 MediaPipe + ViT 摄像头实时识别情绪
2. 稳定 ≥ 1.5 秒后点 **AI 智能诊断**
3. 系统自动调 Kimi（无 Key 走本地模板）生成关怀建议
4. 自动用 pyttsx3 中文 TTS 朗读（可在 `🔊 语音朗读建议` 复选框关闭）
5. **取消诊断** / **停止朗读** 按钮 100~300ms 内收声

### AI 智能问答

| 输入方式 | 默认引擎 |
| --- | --- |
| 打字 | 直接发送 |
| 麦克风 | Vosk 离线（`model/vosk-model-small-cn-0.22/`） |

回复**默认自动朗读**（`AI 回复自动语音播报` 复选框开启时），可点 `停止朗读` 立刻打断。

---

## 🩺 健康监测子系统

> 📄 详细文档：[健康大屏](md/HEALTH_DASHBOARD_README.md) · [智能告警系统](md/ALARM_SYSTEM_README.md) · [华为手环 9 接入指南](md/HUAWEI_BAND_GUIDE.md) · [飞书机器人配置指南](md/飞书机器人配置指南.md)

### 健康指标大屏

从主界面顶部「健康大屏」按钮打开（或 `python test/test_health_dashboard.py` 独立测试）：

- **实时指标**：心率 ❤️ / 血氧 🫁 / 血压 🩸 / 步数 👟 / 体温 🌡️ / 电量 🔋，QTimer 每 2 秒自动刷新；顶部显示设备连接状态与数据更新时间戳
- **ECharts 趋势图**：点击任意指标卡片弹出，支持 24 小时 / 7 天 / 30 天切换，渐变面积图、异常区间高亮、阈值标记线（依赖 `PyQt6-WebEngine`，已包含在 requirements.txt；首次打开需联网加载 CDN）
- **数据导出**：一键导出 UTF-8 编码的 CSV 健康报告，Excel 可直接打开
- **数据来源**：默认由 `HealthDataSimulator` 模拟生成（正态分布随机波动，保留约 24 小时 / 43200 条历史）；接入真实设备见下文

### 智能健康告警（AlarmManager）

大屏内置两级告警，异常时红色高亮并播放 `alarm.mp3`：

| 指标 | 警告 WARNING 🟡 | 危险 CRITICAL 🔴 |
| --- | --- | --- |
| 心率 | - | < 60 或 > 100 bpm |
| 血氧 | < 95% | - |
| 收缩压 | - | < 90 或 > 140 mmHg |
| 体温 | - | > 37.5 ℃ |
| 电量 | - | < 20% |

- **危险级告警**自动弹出置顶告警弹窗（含时间 / 当前值 / 健康建议）
- **告警记录面板**（大屏底部）：实时显示最近 10 条记录 + 总计/警告/危险统计 + 静音开关 + 一键清空；内存中最多保留 500 条
- **冷却机制**：同类告警 5 秒内不重复触发，避免声音轰炸
- 独立测试：`python -m pytest test/test_alarm_system.py`

### 华为手环 9 真实设备接入（可选）

默认运行在模拟器模式；接入真实手环的数据流架构：

```
华为手环 9 → 蓝牙 → 手机 Gadgetbridge（开源 App）→ MQTT Broker（Mosquitto）→ 本项目
```

1. Android 手机安装 [Gadgetbridge](https://gadgetbridge.org/)（免费开源），配对手环；心率/血氧/步数/体温/电量均可直采
2. 电脑安装并启动 [Mosquitto](https://mosquitto.org/)（默认端口 1883）
3. 在 Gadgetbridge 设置中启用 MQTT 推送，指向电脑 IP
4. 修改 `deidao/device_config.json`：`mode` 由 `simulator` 改为 `huawei_band`，填入 MQTT 地址与用户档案（年龄 / 性别 / 静息心率）
5. 按「Mosquitto → Gadgetbridge → 本项目」顺序启动，健康大屏右上角显示「华为手环 - 已连接」即接入成功

**血压估算算法**：手环 9 不支持血压直测，系统基于心率变异性（HRV）估算——`收缩压 = 基础收缩压 + (当前心率 − 静息心率) × 0.5`（舒张压系数 0.2），基础值按年龄段与性别查表。⚠️ 估算值仅供参考，不能替代医疗设备测量。

### 飞书机器人推送配置

系统检测到跌倒后，自动向飞书群推送**交互卡片**（发生时间 / 录像存档文件名 / 系统状态）。配置步骤：

1. 飞书创建群聊 → 群设置 → **群机器人** → 添加**自定义机器人**，复制生成的 Webhook 地址（注意保密，泄露后任何人都能向群里发消息）
2. 写入项目根目录 `.env`：`FEISHU_WEBHOOK=https://open.feishu.cn/open-apis/bot/v2/hook/<token>`
3. 重启程序生效。配置读取优先级：`系统环境变量 > .env 文件 > 代码默认值`（由 `deidao/database.py` 统一注入）

> 未配置 Webhook 时系统自动跳过推送并在日志提示，不影响其他功能；内置推送频率控制（`last_feishu_time`）防止短时间内重复推送。

---

## 📐 告警与 TTS 链路

```
跌倒检测线程
  └─ "跌倒中" 连续 N 帧
      ├─ 播放 alarm.mp3
      ├─ 写入 result/xxx.mp4（含前置缓冲区）
      ├─ INSERT INTO alarm_logs
      └─ POST 飞书 webhook 交互卡片
                  ↓
        EmotionDetectionDialog._speak_async()
                  ↓
        pyttsx3.init (sapi5) → say/iterate → stop → endLoop → del
        (每个 worker 一次性 init→销毁，SAPI5 不允许 engine 复用)
                  ↓
        主线程点击取消 → set _stop_flag → 100~300ms 内 engine.stop() 收声
```

---

## 🧷 TTS 工程要点

经过实际踩坑，本项目 TTS worker 遵循以下硬性约束（详见 `deidao/dialogs.py`）：

1. **SAPI5 backend 不允许 engine 复用**：第二次 `engine.say()` 经常哑火。每个 worker 一次性 init → 用 → `del`。
2. **绝对不能用 `engine.runAndWait()` 硬阻塞**：那样按"停止"也要等当前句放完。改用 `startLoop(False) + iterate()`，句间轮询 `_stop_flag`，命中即 `engine.stop()`。
3. **绝不在主线程 touch engine**：sapi5 是 apartment-threaded COM，跨 apartment 调用会触发 marshalling 死锁。
4. **主线程只设标志位 `self._stop_flag = True`**，所有 stop/endLoop/del 都在 worker 自己的线程里同步完成。

---

## 🛠️ 常见问题

| 现象 | 排查 |
| --- | --- |
| 主程序启动即 "导入 PyQt6 失败" | `pip install --upgrade PyQt6` |
| 第一次朗读 OK，第二次无声 | 已修复，确保拉到最新 `dialogs.py` |
| AI 弹窗不朗读 | 看 `lbl_tts_status` 状态栏：显示 `TTS 不可用: ...` 则按错误处理 |
| 点"停止朗读"按钮像没生效 | 已用 `startLoop+iterate`，延迟 100~300ms；如仍卡死，重启程序即可恢复 |
| MySQL 连不上 | 检查 `localhost:3306` 服务，确认账号密码与 `.env` |
| Vosk 识别没反应 | 确认 `model/vosk-model-small-cn-0.22/` 存在且为完整解压目录 |
| Kimi 不回复 | 没 Key 自动回退本地规则模板；想看云端回复请配 `KIMI_API_KEY` |
| `torch_cuda.dll 找不到 cudart` | `main.py` 已自动注入 `torch/lib` 到 PATH；保持最新代码即可 |

---

## 🧭 后续可扩展方向

- **加权打分机制**：给不同特征分配权重，总分超过阈值才报警（当前使用共识机制）
- **机器学习分类器**：收集大量数据训练 SVM/Random Forest，替代手工规则
- **时序模型**：使用 LSTM/GRU 学习动作时间模式，更准确区分跌倒和日常活动
- **告警增强**：微信/短信推送家属、告警阈值自定义、告警历史导出 CSV/PDF、告警趋势分析、更多告警级别
- **硬件联动**：通过 MQTT / TCP 接入智能手环、毫米波雷达
- **多摄像头 / 多人**：将检测逻辑并发化，扩展至 N 路视频
- **云端数据库**：MySQL → 阿里云 / 腾讯云 RDS，支持跨地域家属查看
- **模型升级**：YOLOv8 Pose、RTMPose 等端到姿态估计
- **语音对讲**：集成 WebRTC，跌倒后自动外呼家属
- **本地 LLM**：在 `answer_question()` 接入 Qwen2 / ChatGLM3，不再依赖网络

---

## 🤝 贡献

欢迎 PR / Issue：

1. Fork → 新建分支 (`git checkout -b feature/xxx`)
2. 提交 (`git commit -m "feat: ..."`)
3. 推送 (`git push origin feature/xxx`)
4. 发起 Pull Request

代码风格：PEP 8；关键模块（`dialogs.py` / `detector.py`）请跑 `pytest test/`。

---

## 📄 许可证

本项目基于 **MIT License** 发布，详见 [LICENSE](LICENSE)。

> ⚠️ 项目内含明文 MySQL / 飞书 Webhook / Kimi Key 占位，**仅供学习演示**。
> 正式部署请使用环境变量 / 密钥管理服务（如 HashiCorp Vault）。
