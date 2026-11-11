"""
面部情绪实时检测 - 性能优化版
用法:
    python app.py                  # 自动检测摄像头
    python app.py 0                # 使用摄像头 0
    python app.py video.mp4        # 使用视频文件
    python app.py video.mp4 --batch 10  # 视频批量推理(每批10帧)
    python app.py --imgsz 416      # 降低推理分辨率提升速度
    python app.py --conf 0.3       # 降低置信度阈值
快捷键:
    ESC   - 退出
    T     - 提高置信度阈值 (+0.05)
    G     - 降低置信度阈值 (-0.05)
    S     - 截图保存
"""
import os
import sys
import time
import argparse
from collections import deque
from pathlib import Path

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


def find_available_camera(max_index=5):
    """尝试找到可用的摄像头"""
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"[OK] 找到可用摄像头: 索引 {i}")
            cap.release()
            return i
        cap.release()
    return None


def parse_args():
    parser = argparse.ArgumentParser(description="面部情绪实时检测")
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
    return parser.parse_args()


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

    def __init__(self, model_path, imgsz=640, conf=0.5):
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
            self.emotion_stats[name] = self.emotion_stats.get(name, 0) + 1

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

    def process_frame(self, frame):
        """处理单帧 - 核心推理（优化版：跳过灰度转换）"""
        # 直接用彩色 BGR 帧推理，不再转灰度再合并
        # 原代码: gray = cv2.cvtColor(frame, BGR2GRAY); gray_3d = merge([gray]*3)
        # 优化: 直接传 frame，YOLO 内部会处理
        results = self.model(frame, imgsz=self.imgsz, conf=self.conf, verbose=False)
        result = results[0]

        # 更新统计
        self._update_stats(result)

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
            annotated = result.plot()
            self._update_fps()
            fps = self._get_fps()
            self._draw_info(annotated, fps)
            annotated_list.append(annotated)

        return annotated_list


def main():
    args = parse_args()

    # 确定视频源
    source = None
    if args.source is not None:
        if args.source.isdigit():
            source = int(args.source)
        elif os.path.exists(args.source):
            source = args.source
        else:
            print(f"[FAIL] 无效参数: {args.source}")
            print("用法: python app.py [摄像头索引|视频文件路径]")
            sys.exit(1)
    else:
        # 自动检测摄像头
        print("[INFO] 自动检测摄像头...")
        cam_idx = find_available_camera()
        if cam_idx is not None:
            source = cam_idx
        else:
            print("[FAIL] 未找到可用摄像头")
            print("解决方案:")
            print("  - 使用视频文件: python app.py video.mp4")
            print("  - 指定摄像头索引: python app.py 1")
            sys.exit(1)

    # 初始化检测器
    detector = EmotionDetector(
        model_path=args.model,
        imgsz=args.imgsz,
        conf=args.conf,
    )

    # 打开视频源
    is_video_file = isinstance(source, str)
    if is_video_file:
        print(f"[INFO] 打开视频文件: {source}")
    else:
        print(f"[INFO] 打开摄像头 {source}...")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        if is_video_file:
            print(f"[FAIL] 无法打开视频文件: {source}")
        else:
            print("[FAIL] 无法打开摄像头")
            print("可能原因:")
            print("  1. 摄像头被其他程序占用")
            print("  2. 摄像头驱动未安装")
            print("解决方案:")
            print("  - python app.py video.mp4")
            print("  - python app.py 1")
        sys.exit(1)

    # 获取视频信息
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if is_video_file and total_frames > 0:
        print(f"[INFO] 视频: {total_frames} 帧, {fps_video:.1f} FPS")

    # 创建输出目录
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"[INFO] 推理分辨率: {args.imgsz}x{args.imgsz}")
    print(f"[INFO] 置信度阈值: {args.conf}")
    if args.batch > 0:
        print(f"[INFO] 批量推理: 每批 {args.batch} 帧")
    print("[INFO] 开始检测, 按 ESC 退出, T/G 调节置信度, S 截图")

    # 批量推理缓冲
    batch_buffer = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if is_video_file:
                    if batch_buffer:
                        # 处理剩余帧
                        results = detector.process_batch(batch_buffer)
                        for r in results:
                            cv2.imshow("Emotion Detection", r)
                            if cv2.waitKey(1) == 27:
                                break
                        batch_buffer.clear()
                    # 视频播放完毕，从头开始
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    print("[INFO] 视频播放完毕，重新开始...")
                    continue
                else:
                    print("[FAIL] 无法读取摄像头帧")
                    break

            if args.batch > 0 and is_video_file:
                # 批量推理模式
                batch_buffer.append(frame)
                if len(batch_buffer) >= args.batch:
                    results = detector.process_batch(batch_buffer)
                    for r in results:
                        cv2.imshow("Emotion Detection", r)
                        key = cv2.waitKey(1)
                        if key == 27:
                            break
                        elif key == ord("s") or key == ord("S"):
                            save_path = os.path.join(
                                args.save_dir,
                                f"screenshot_{detector.frame_count:06d}.jpg",
                            )
                            cv2.imwrite(save_path, r)
                            print(f"[OK] 截图保存: {save_path}")
                    batch_buffer.clear()
            else:
                # 单帧推理模式（摄像头 / 非批量视频）
                annotated = detector.process_frame(frame)
                cv2.imshow("Emotion Detection", annotated)

                key = cv2.waitKey(1)
                if key == 27:  # ESC
                    break
                elif key == ord("t") or key == ord("T"):
                    detector.update_confidence(0.05)
                elif key == ord("g") or key == ord("G"):
                    detector.update_confidence(-0.05)
                elif key == ord("s") or key == ord("S"):
                    save_path = os.path.join(
                        args.save_dir,
                        f"screenshot_{detector.frame_count:06d}.jpg",
                    )
                    cv2.imwrite(save_path, annotated)
                    print(f"[OK] 截图保存: {save_path}")

    except KeyboardInterrupt:
        print("\n[INFO] 用户中断")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    # 打印最终统计
    print("\n" + "=" * 40)
    print("[INFO] 检测结束")
    print(f"  总帧数: {detector.frame_count}")
    print(f"  总检测: {detector.detect_count}")
    if detector.emotion_stats:
        total = sum(detector.emotion_stats.values())
        print("  情绪分布:")
        for name, count in sorted(
            detector.emotion_stats.items(), key=lambda x: -x[1]
        ):
            cn = detector.EMOTION_CN.get(name, name)
            print(f"    {cn} ({name}): {count} ({count/total*100:.1f}%)")
    print("=" * 40)


if __name__ == "__main__":
    # Windows 中文输出兼容
    if sys.platform == "win32":
        os.environ["PYTHONIOENCODING"] = "utf-8"
    main()
