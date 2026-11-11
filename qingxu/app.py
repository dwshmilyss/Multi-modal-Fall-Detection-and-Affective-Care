"""
面部情绪实时检测 - 功能增强版
用法:
    python app.py                          # 自动检测摄像头
    python app.py 0                        # 使用摄像头 0
    python app.py video.mp4                # 使用视频文件
    python app.py video.mp4 --batch 10     # 视频批量推理(每批10帧)
    python app.py --imgsz 416              # 降低推理分辨率提升速度
    python app.py --conf 0.3               # 降低置信度阈值
    python app.py --config config.yaml     # 使用配置文件
    python app.py --no-log                 # 禁用日志记录
    python app.py --no-record              # 禁用录像功能

快捷键:
    ESC   - 退出
    T     - 提高置信度阈值 (+0.05)
    G     - 降低置信度阈值 (-0.05)
    S     - 截图保存
    R     - 开始/停止录像
    H     - 显示/隐藏历史记录面板
"""
import os
import sys
import time
import csv
import json
import argparse
from collections import deque
from pathlib import Path
from datetime import datetime

# 确保 onnxruntime 能找到 cuDNN/cuBLAS DLL (GPU 加速必需)
if sys.platform == "win32":
    _venv = Path(sys.executable).parent.parent
    _cuda_paths = [
        str(_venv / "Lib" / "site-packages" / "nvidia" / "cudnn" / "bin"),
        str(_venv / "Lib" / "site-packages" / "nvidia" / "cublas" / "bin"),
        str(_venv / "Lib" / "site-packages" / "nvidia" / "cuda_nvrtc" / "bin"),
    ]
    os.environ["PATH"] = ";".join(_cuda_paths) + ";" + os.environ.get("PATH", "")

import cv2
import numpy as np
from ultralytics import YOLO

# 可选依赖
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


# ============== 配置文件加载 ==============

def load_config(config_path):
    """加载 YAML 配置文件"""
    if not HAS_YAML:
        print("[WARN] PyYAML 未安装，使用默认配置")
        return {}
    
    if not os.path.exists(config_path):
        print(f"[WARN] 配置文件不存在: {config_path}，使用默认配置")
        return {}
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"[OK] 加载配置文件: {config_path}")
        return config if config else {}
    except Exception as e:
        print(f"[WARN] 配置文件加载失败: {e}，使用默认配置")
        return {}


# ============== 情绪历史记录器 ==============

class EmotionHistoryRecorder:
    """情绪历史记录器 - 每 N 秒统计一次情绪分布"""
    
    def __init__(self, interval=5, max_records=288):
        """
        Args:
            interval: 统计间隔（秒）
            max_records: 最大记录数（默认 288 = 24小时 * 12次/小时）
        """
        self.interval = interval
        self.max_records = max_records
        self.history = []
        self.current_emotions = {}
        self.last_record_time = time.time()
    
    def update(self, emotion_name):
        """更新当前情绪计数"""
        self.current_emotions[emotion_name] = self.current_emotions.get(emotion_name, 0) + 1
    
    def record_if_needed(self):
        """如果到达间隔时间，记录一次统计"""
        now = time.time()
        if now - self.last_record_time >= self.interval:
            self.record()
            self.last_record_time = now
    
    def record(self):
        """记录当前情绪分布"""
        total = sum(self.current_emotions.values())
        if total == 0:
            return
        
        record = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_faces': total,
        }
        
        # 计算每种情绪的百分比
        for emotion, count in self.current_emotions.items():
            record[emotion] = round(count / total * 100, 1)
        
        self.history.append(record)
        
        # 限制记录数量
        if len(self.history) > self.max_records:
            self.history = self.history[-self.max_records:]
        
        # 重置当前计数
        self.current_emotions = {}
    
    def save(self, filepath):
        """保存历史记录到 JSON"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
            print(f"[OK] 历史记录已保存: {filepath} ({len(self.history)} 条记录)")
        except Exception as e:
            print(f"[FAIL] 保存历史记录失败: {e}")
    
    def load(self, filepath):
        """从 JSON 加载历史记录"""
        if not os.path.exists(filepath):
            return
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                self.history = json.load(f)
            print(f"[OK] 加载历史记录: {filepath} ({len(self.history)} 条记录)")
        except Exception as e:
            print(f"[WARN] 加载历史记录失败: {e}")
    
    def generate_trend_chart(self, output_path):
        """使用 Plotly 生成情绪趋势图"""
        if not HAS_PLOTLY:
            print("[WARN] Plotly 未安装，无法生成趋势图")
            return False
        
        if len(self.history) < 2:
            print("[WARN] 历史记录不足，无法生成趋势图")
            return False
        
        try:
            # 项目配色方案
            COLORS = {
                'bg': '#0F172A',
                'card': '#1E293B',
                'accent': '#38BDF8',
                'happy': '#10B981',
                'sad': '#EF4444',
                'angry': '#F59E0B',
                'fearful': '#8B5CF6',
                'neutral': '#94A3B8',
                'text': '#F8FAFC',
                'grid': '#334155',
            }
            
            # 提取数据
            times = [r['timestamp'] for r in self.history]
            emotions = ['Happy', 'Sad', 'Angry', 'Fearful', 'Neutral']
            emotion_colors = [COLORS['happy'], COLORS['sad'], COLORS['angry'], COLORS['fearful'], COLORS['neutral']]
            emotion_cn = {'Happy': '开心', 'Sad': '悲伤', 'Angry': '愤怒', 'Fearful': '恐惧', 'Neutral': '平静'}
            
            fig = go.Figure()
            
            for emotion, color in zip(emotions, emotion_colors):
                values = [r.get(emotion, 0) for r in self.history]
                r_val = int(color[1:3], 16)
                g_val = int(color[3:5], 16)
                b_val = int(color[5:7], 16)
                fill_color = f'rgba({r_val}, {g_val}, {b_val}, 0.15)'
                
                fig.add_trace(go.Scatter(
                    x=times,
                    y=values,
                    name=f"{emotion} ({emotion_cn[emotion]})",
                    mode='lines',
                    line=dict(width=2.5, color=color),
                    fill='tozeroy',
                    fillcolor=fill_color,
                    opacity=0.9,
                ))
            
            fig.update_layout(
                title=dict(
                    text='Emotion Trend Analysis - 情绪趋势分析',
                    font=dict(size=22, color=COLORS['text'], family='Segoe UI'),
                    x=0.5,
                ),
                plot_bgcolor=COLORS['bg'],
                paper_bgcolor=COLORS['bg'],
                font=dict(color=COLORS['text'], family='Segoe UI'),
                legend=dict(
                    bgcolor='rgba(30, 41, 59, 0.8)',
                    bordercolor=COLORS['grid'],
                    borderwidth=1,
                    font=dict(size=12),
                    x=0.02,
                    y=0.98,
                ),
                xaxis=dict(
                    gridcolor=COLORS['grid'],
                    linecolor=COLORS['grid'],
                    title='Time',
                ),
                yaxis=dict(
                    gridcolor=COLORS['grid'],
                    linecolor=COLORS['grid'],
                    title='Percentage (%)',
                    range=[0, 100],
                ),
                margin=dict(l=60, r=30, t=80, b=60),
                height=500,
                hovermode='x unified',
                hoverlabel=dict(
                    bgcolor=COLORS['card'],
                    bordercolor=COLORS['accent'],
                    font=dict(color=COLORS['text']),
                ),
            )
            
            fig.write_image(output_path, scale=2)
            print(f"[OK] 情绪趋势图已保存: {output_path}")
            return True
            
        except Exception as e:
            print(f"[FAIL] 生成趋势图失败: {e}")
            return False


# ============== CSV 日志记录器 ==============

class EmotionLogger:
    """CSV 日志记录器"""
    
    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成日志文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = self.log_dir / f"emotion_log_{timestamp}.csv"
        
        # 写入 CSV 头
        with open(self.log_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'emotion', 'confidence', 'face_count'])
        
        print(f"[OK] 日志文件已创建: {self.log_file}")
    
    def log(self, emotion, confidence, face_count):
        """记录一条检测结果"""
        try:
            with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    emotion,
                    f"{confidence:.2f}",
                    face_count,
                ])
        except Exception as e:
            print(f"[WARN] 日志写入失败: {e}")


# ============== 录像管理器 ==============

class VideoRecorder:
    """录像管理器"""
    
    def __init__(self, output_dir, fps=30, codec='mp4v'):
        self.output_dir = Path(output_dir) / "recordings"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.codec = codec
        self.writer = None
        self.recording = False
        self.record_file = None
    
    def start(self, frame_size):
        """开始录像"""
        if self.recording:
            return
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.record_file = self.output_dir / f"recording_{timestamp}.mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self.writer = cv2.VideoWriter(
            str(self.record_file),
            fourcc,
            self.fps,
            (frame_size[1], frame_size[0]),  # (width, height)
        )
        
        if self.writer.isOpened():
            self.recording = True
            print(f"[OK] 开始录像: {self.record_file}")
        else:
            print("[FAIL] 录像启动失败")
            self.writer = None
    
    def write(self, frame):
        """写入一帧"""
        if self.recording and self.writer is not None:
            self.writer.write(frame)
    
    def stop(self):
        """停止录像"""
        if not self.recording:
            return
        
        self.recording = False
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            print(f"[OK] 录像已保存: {self.record_file}")
    
    def is_recording(self):
        """是否正在录像"""
        return self.recording


# ============== 情绪检测器 ==============

class EmotionDetector:
    """面部情绪检测器"""

    # 情绪标签映射 (英文 -> 中文)
    EMOTION_CN = {
        "Angry": "愤怒",
        "Fearful": "恐惧",
        "Happy": "开心",
        "Neutral": "平静",
        "Sad": "悲伤",
    }

    def __init__(self, model_path, imgsz=640, conf=0.5, enable_log=True, enable_record=True):
        if not os.path.exists(model_path):
            print(f"[FAIL] 模型文件不存在: {model_path}")
            print("请确保 best.onnx 文件在当前目录下")
            sys.exit(1)

        print(f"[INFO] 加载模型: {model_path}")
        self.model = YOLO(model_path)
        self.imgsz = imgsz
        self.conf = conf

        # FPS 统计
        self.fps_buffer = deque(maxlen=30)
        self.last_time = time.perf_counter()

        # 统计信息
        self.frame_count = 0
        self.detect_count = 0
        self.emotion_stats = {}
        
        # 情绪历史记录器
        self.history_recorder = EmotionHistoryRecorder(interval=5)
        
        # CSV 日志记录器
        self.logger = None
        if enable_log:
            self.logger = EmotionLogger("output/logs")
        
        # 录像管理器
        self.recorder = None
        if enable_record:
            self.recorder = VideoRecorder("output")
        
        # 显示选项
        self.show_history_panel = False

    def update_confidence(self, delta):
        """调整置信度阈值"""
        self.conf = max(0.1, min(0.95, self.conf + delta))
        print(f"[INFO] 置信度阈值: {self.conf:.2f}")

    def _update_fps(self):
        """更新 FPS 统计"""
        now = time.perf_counter()
        elapsed = now - self.last_time
        self.last_time = now
        if elapsed > 0:
            self.fps_buffer.append(1.0 / elapsed)
        self.frame_count += 1

    def _get_fps(self):
        """获取当前 FPS"""
        if not self.fps_buffer:
            return 0.0
        return sum(self.fps_buffer) / len(self.fps_buffer)

    def _update_stats(self, result):
        """更新情绪统计"""
        if result.boxes is None or len(result.boxes) == 0:
            return
        self.detect_count += len(result.boxes)
        for box in result.boxes:
            cls_id = int(box.cls[0])
            name = result.names.get(cls_id, "Unknown")
            conf = float(box.conf[0])
            
            # 更新统计
            self.emotion_stats[name] = self.emotion_stats.get(name, 0) + 1
            
            # 更新历史记录
            self.history_recorder.update(name)
            
            # 记录日志
            if self.logger:
                self.logger.log(name, conf, len(result.boxes))

    def _get_dominant_emotion(self):
        """获取主导情绪"""
        if not self.emotion_stats:
            return "Unknown", 0
        dominant = max(self.emotion_stats.items(), key=lambda x: x[1])
        total = sum(self.emotion_stats.values())
        percentage = dominant[1] / total * 100 if total > 0 else 0
        return dominant[0], percentage

    def _get_emotion_stability(self):
        """计算情绪稳定度（最近 10 次检测中主导情绪的变化频率）"""
        if len(self.history_recorder.history) < 2:
            return 100.0
        
        # 取最近 10 条记录
        recent = self.history_recorder.history[-10:]
        dominant_emotions = []
        
        for record in recent:
            emotions = {k: v for k, v in record.items() if k not in ['timestamp', 'total_faces']}
            if emotions:
                dominant = max(emotions.items(), key=lambda x: x[1])
                dominant_emotions.append(dominant[0])
        
        if len(dominant_emotions) < 2:
            return 100.0
        
        # 计算变化次数
        changes = sum(1 for i in range(1, len(dominant_emotions)) if dominant_emotions[i] != dominant_emotions[i-1])
        stability = (1 - changes / (len(dominant_emotions) - 1)) * 100
        
        return stability

    def _draw_info(self, frame, fps):
        """在画面上绘制 FPS 和统计信息"""
        h, w = frame.shape[:2]

        # FPS (左上角)
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(
            frame, fps_text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
        )

        # 置信度阈值
        conf_text = f"Conf: {self.conf:.2f} (T+/G-)"
        cv2.putText(
            frame, conf_text, (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1,
        )

        # 检测统计 (右上角)
        stats_lines = [
            f"Frames: {self.frame_count}",
            f"Detections: {self.detect_count}",
        ]
        for i, line in enumerate(stats_lines):
            cv2.putText(
                frame, line, (w - 200, 30 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
            )

        # 情绪分布 (右侧中部)
        if self.emotion_stats:
            total = sum(self.emotion_stats.values())
            y_offset = 100
            for name, count in sorted(
                self.emotion_stats.items(), key=lambda x: -x[1]
            ):
                cn_name = self.EMOTION_CN.get(name, name)
                pct = count / total * 100
                bar_text = f"{cn_name}: {pct:.0f}%"
                cv2.putText(
                    frame, bar_text, (w - 180, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1,
                )
                y_offset += 20
        
        # 主导情绪 (左下角)
        dominant_emotion, dominant_pct = self._get_dominant_emotion()
        if dominant_emotion != "Unknown":
            cn_name = self.EMOTION_CN.get(dominant_emotion, dominant_emotion)
            dominant_text = f"Dominant: {cn_name} ({dominant_pct:.0f}%)"
            cv2.putText(
                frame, dominant_text, (10, h - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
            )
        
        # 情绪稳定度
        stability = self._get_emotion_stability()
        stability_text = f"Stability: {stability:.0f}%"
        cv2.putText(
            frame, stability_text, (10, h - 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 100), 1,
        )
        
        # 录像指示 (右上角)
        if self.recorder and self.recorder.is_recording():
            cv2.circle(frame, (w - 30, 30), 10, (0, 0, 255), -1)
            cv2.putText(
                frame, "REC", (w - 70, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
            )

    def process_frame(self, frame):
        """处理单帧 - 核心推理（优化版：跳过灰度转换）"""
        # 直接用彩色 BGR 帧推理，不再转灰度再合并
        # 原代码: gray = cv2.cvtColor(frame, BGR2GRAY); gray_3d = merge([gray]*3)
        # 优化: 直接传 frame，YOLO 内部会处理
        results = self.model(frame, imgsz=self.imgsz, conf=self.conf, verbose=False)
        result = results[0]

        # 更新统计
        self._update_stats(result)
        
        # 更新历史记录
        self.history_recorder.record_if_needed()
        
        # 录像
        if self.recorder and self.recorder.is_recording():
            self.recorder.write(frame)

        # 绘制标注框
        annotated = result.plot()

        # 计算并绘制 FPS
        self._update_fps()
        fps = self._get_fps()
        self._draw_info(annotated, fps)

        return annotated

    def process_batch(self, frames):
        """批量处理多帧（适合视频文件，提升吞吐量）"""
        if not frames:
            return []

        # 将所有帧打包成一个批次推理
        # YOLO 支持 list[ndarray] 输入
        results = self.model(
            frames, imgsz=self.imgsz, conf=self.conf, verbose=False
        )

        annotated_list = []
        for frame, result in zip(frames, results):
            self._update_stats(result)
            self.history_recorder.record_if_needed()
            annotated = result.plot()
            self._update_fps()
            fps = self._get_fps()
            self._draw_info(annotated, fps)
            annotated_list.append(annotated)

        return annotated_list
    
    def save_history(self, output_dir):
        """保存历史记录和生成图表"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 保存 JSON
        json_path = output_path / "emotion_history.json"
        self.history_recorder.save(str(json_path))
        
        # 生成趋势图
        chart_path = output_path / "emotion_trend.png"
        self.history_recorder.generate_trend_chart(str(chart_path))
        
        # 生成情绪分布饼图
        self._generate_pie_chart(str(output_path / "emotion_distribution.png"))
    
    def _generate_pie_chart(self, output_path):
        """生成情绪分布饼图"""
        if not HAS_PLOTLY or not self.emotion_stats:
            return
        
        try:
            COLORS = {
                'bg': '#0F172A',
                'happy': '#10B981',
                'sad': '#EF4444',
                'angry': '#F59E0B',
                'fearful': '#8B5CF6',
                'neutral': '#94A3B8',
                'text': '#F8FAFC',
            }
            
            emotion_colors_map = {
                'Happy': COLORS['happy'],
                'Sad': COLORS['sad'],
                'Angry': COLORS['angry'],
                'Fearful': COLORS['fearful'],
                'Neutral': COLORS['neutral'],
            }
            
            labels = []
            values = []
            colors = []
            
            for emotion, count in sorted(self.emotion_stats.items(), key=lambda x: -x[1]):
                cn_name = self.EMOTION_CN.get(emotion, emotion)
                labels.append(f"{cn_name} ({emotion})")
                values.append(count)
                colors.append(emotion_colors_map.get(emotion, '#999999'))
            
            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                marker=dict(colors=colors),
                textinfo='label+percent',
                textfont=dict(color=COLORS['text'], size=14),
                hole=0.4,
            )])
            
            fig.update_layout(
                title=dict(
                    text='Emotion Distribution - 情绪分布',
                    font=dict(size=20, color=COLORS['text']),
                    x=0.5,
                ),
                paper_bgcolor=COLORS['bg'],
                font=dict(color=COLORS['text']),
                height=500,
            )
            
            fig.write_image(output_path, scale=2)
            print(f"[OK] 情绪分布图已保存: {output_path}")
            
        except Exception as e:
            print(f"[FAIL] 生成饼图失败: {e}")


# ============== 命令行参数解析 ==============

def parse_args():
    parser = argparse.ArgumentParser(description="面部情绪实时检测 - 功能增强版")
    parser.add_argument(
        "source",
        nargs="?",
        default=None,
        help="摄像头索引或视频文件路径 (默认: 自动检测摄像头)",
    )
    parser.add_argument(
        "--model",
        default="best.onnx",
        help="模型文件路径 (默认: best.onnx)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        choices=[320, 416, 512, 640],
        help="推理分辨率, 越低越快 (默认: 640)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="置信度阈值 0-1 (默认: 0.5)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=0,
        help="视频批量推理帧数, 0=禁用 (默认: 0)",
    )
    parser.add_argument(
        "--save-dir",
        default="output",
        help="截图/录像保存目录 (默认: output)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="配置文件路径 (YAML 格式)",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="禁用 CSV 日志记录",
    )
    parser.add_argument(
        "--no-record",
        action="store_true",
        help="禁用录像功能",
    )
    parser.add_argument(
        "--history-interval",
        type=int,
        default=5,
        help="情绪历史记录间隔（秒）(默认: 5)",
    )
    return parser.parse_args()


# ============== 主函数 ==============

def main():
    args = parse_args()
    
    # 加载配置文件（如果指定）
    config = {}
    if args.config:
        config = load_config(args.config)
    
    # 合并配置（命令行参数 > 配置文件 > 默认值）
    model_path = args.model if args.model != "best.onnx" else config.get('model', 'best.onnx')
    imgsz = args.imgsz if args.imgsz != 640 else config.get('imgsz', 640)
    conf = args.conf if args.conf != 0.5 else config.get('conf', 0.5)
    save_dir = args.save_dir if args.save_dir != "output" else config.get('save_dir', 'output')
    enable_log = not args.no_log and config.get('auto_log', True)
    enable_record = not args.no_record and config.get('enable_record', True)
    history_interval = args.history_interval if args.history_interval != 5 else config.get('history_interval', 5)

    # 确定视频源
    source = None
    if args.source is not None:
        if args.source.isdigit():
            source = int(args.source)
        elif os.path.exists(args.source):
            source = args.source
        else:
            print(f"[FAIL] 视频源不存在: {args.source}")
            sys.exit(1)
    else:
        source = find_available_camera()
        if source is None:
            print("[FAIL] 未找到可用摄像头")
            print("用法: python app.py <视频文件路径>")
            sys.exit(1)

    # 创建检测器
    detector = EmotionDetector(
        model_path=model_path,
        imgsz=imgsz,
        conf=conf,
        enable_log=enable_log,
        enable_record=enable_record,
    )
    detector.history_recorder.interval = history_interval

    # 打开视频源
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[FAIL] 无法打开视频源: {source}")
        sys.exit(1)

    print(f"[OK] 成功连接视频源: {source}")
    print(f"[INFO] 分辨率: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print(f"[INFO] 快捷键: ESC=退出, T/G=置信度, S=截图, R=录像, H=历史面板")

    # 批量推理模式
    batch_size = args.batch
    batch_frames = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] 视频结束")
                break

            if batch_size > 0:
                batch_frames.append(frame)
                if len(batch_frames) >= batch_size:
                    results = detector.process_batch(batch_frames)
                    for result in results:
                        cv2.imshow("Emotion Detection", result)
                    batch_frames = []
            else:
                result = detector.process_frame(frame)
                cv2.imshow("Emotion Detection", result)

            # 处理按键
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('t') or key == ord('T'):
                detector.update_confidence(0.05)
            elif key == ord('g') or key == ord('G'):
                detector.update_confidence(-0.05)
            elif key == ord('s') or key == ord('S'):
                # 截图
                screenshot_dir = Path(save_dir) / "screenshots"
                screenshot_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                screenshot_path = screenshot_dir / f"screenshot_{timestamp}.png"
                cv2.imwrite(str(screenshot_path), frame)
                print(f"[OK] 截图已保存: {screenshot_path}")
            elif key == ord('r') or key == ord('R'):
                # 录像
                if detector.recorder:
                    if detector.recorder.is_recording():
                        detector.recorder.stop()
                    else:
                        detector.recorder.start(frame.shape)
            elif key == ord('h') or key == ord('H'):
                # 切换历史面板显示
                detector.show_history_panel = not detector.show_history_panel
                print(f"[INFO] 历史面板: {'开启' if detector.show_history_panel else '关闭'}")

    except KeyboardInterrupt:
        print("\n[INFO] 用户中断")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        # 停止录像
        if detector.recorder and detector.recorder.is_recording():
            detector.recorder.stop()
        
        # 保存历史记录和生成图表
        print("\n[INFO] 保存统计数据...")
        detector.save_history(save_dir)
        
        # 打印最终统计
        print("\n" + "="*50)
        print(f"[OK] 处理完成: {detector.frame_count} 帧")
        print(f"[OK] 检测到 {detector.detect_count} 个人脸")
        print(f"[OK] 结果保存在: {save_dir}")
        print("="*50)
        
        if detector.emotion_stats:
            total = sum(detector.emotion_stats.values())
            print("\n情绪分布:")
            for name, count in sorted(detector.emotion_stats.items(), key=lambda x: -x[1]):
                cn_name = detector.EMOTION_CN.get(name, name)
                pct = count / total * 100
                print(f"  {cn_name} ({name}): {count} ({pct:.1f}%)")


if __name__ == "__main__":
    # 设置 UTF-8 编码
    if sys.platform == "win32":
        os.environ["PYTHONIOENCODING"] = "utf-8"
    main()
