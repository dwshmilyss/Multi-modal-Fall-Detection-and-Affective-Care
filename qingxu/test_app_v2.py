"""
测试脚本 - 验证方案B所有新功能
"""
import sys
import os
from pathlib import Path
from datetime import datetime

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from app import (
    EmotionDetector, 
    EmotionHistoryRecorder, 
    EmotionLogger, 
    VideoRecorder,
    load_config,
)

def test_config_loading():
    """测试 1: 配置文件加载"""
    print("\n=== 测试 1: 配置文件加载 ===")
    
    # 测试加载存在的配置文件
    config_path = str(Path(__file__).parent / "config.yaml")
    config = load_config(config_path)
    assert 'model' in config, "配置文件应包含 model 字段"
    assert 'imgsz' in config, "配置文件应包含 imgsz 字段"
    print(f"[OK] 配置文件加载成功: {config}")
    
    # 测试加载不存在的配置文件
    config = load_config("nonexistent.yaml")
    assert config == {}, "不存在的配置文件应返回空字典"
    print(f"[OK] 不存在的配置文件处理正确")
    
    print("[PASS] 配置文件加载测试通过")


def test_history_recorder():
    """测试 2: 情绪历史记录器"""
    print("\n=== 测试 2: 情绪历史记录器 ===")
    
    recorder = EmotionHistoryRecorder(interval=1)
    
    # 添加一些情绪数据
    recorder.update("Happy")
    recorder.update("Happy")
    recorder.update("Sad")
    
    # 强制记录
    recorder.record()
    
    assert len(recorder.history) == 1, "应有一条历史记录"
    assert abs(recorder.history[0]['Happy'] - 66.7) < 0.1, f"Happy 应占约 66.7%，实际 {recorder.history[0]['Happy']}"
    assert abs(recorder.history[0]['Sad'] - 33.3) < 0.1, f"Sad 应占约 33.3%，实际 {recorder.history[0]['Sad']}"
    print(f"[OK] 历史记录: {recorder.history[0]}")
    
    # 测试保存和加载
    test_file = "test_history.json"
    recorder.save(test_file)
    assert os.path.exists(test_file), "历史文件应已创建"
    
    recorder2 = EmotionHistoryRecorder()
    recorder2.load(test_file)
    assert len(recorder2.history) == 1, "加载后应有一条记录"
    print(f"[OK] 保存和加载成功")
    
    # 清理
    os.remove(test_file)
    
    print("[PASS] 情绪历史记录器测试通过")


def test_emotion_logger():
    """测试 3: CSV 日志记录器"""
    print("\n=== 测试 3: CSV 日志记录器 ===")
    
    logger = EmotionLogger("test_logs")
    
    # 记录一些数据
    logger.log("Happy", 0.95, 1)
    logger.log("Sad", 0.85, 1)
    logger.log("Happy", 0.90, 2)
    
    # 检查文件是否存在
    log_files = list(Path("test_logs").glob("*.csv"))
    assert len(log_files) == 1, "应创建一个 CSV 文件"
    
    # 读取并验证内容
    with open(log_files[0], 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    assert len(lines) == 4, "应有 1 行表头 + 3 行数据"
    assert "timestamp" in lines[0], "第一行应为表头"
    assert "Happy" in lines[1], "第二行应包含 Happy"
    print(f"[OK] 日志文件内容: {lines[0].strip()}")
    print(f"[OK] 第一条记录: {lines[1].strip()}")
    
    # 清理
    import shutil
    shutil.rmtree("test_logs")
    
    print("[PASS] CSV 日志记录器测试通过")


def test_video_recorder():
    """测试 4: 录像管理器"""
    print("\n=== 测试 4: 录像管理器 ===")
    
    recorder = VideoRecorder("test_output", fps=10)
    
    # 创建测试帧
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # 开始录像
    recorder.start(frame.shape)
    assert recorder.is_recording(), "录像应已开始"
    print(f"[OK] 录像已开始")
    
    # 写入几帧
    for i in range(10):
        recorder.write(frame)
    
    # 停止录像
    recorder.stop()
    assert not recorder.is_recording(), "录像应已停止"
    print(f"[OK] 录像已停止")
    
    # 检查文件
    recording_files = list(Path("test_output/recordings").glob("*.mp4"))
    assert len(recording_files) == 1, "应创建一个录像文件"
    print(f"[OK] 录像文件已创建: {recording_files[0]}")
    
    # 清理
    import shutil
    shutil.rmtree("test_output")
    
    print("[PASS] 录像管理器测试通过")


def test_detector_integration():
    """测试 5: 检测器集成测试"""
    print("\n=== 测试 5: 检测器集成测试 ===")
    
    # 创建检测器（禁用日志和录像以加快测试）
    model_path = str(Path(__file__).parent / "best.onnx")
    detector = EmotionDetector(
        model_path=model_path,
        imgsz=320,
        conf=0.5,
        enable_log=False,
        enable_record=False,
    )
    
    # 创建测试帧
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # 处理一帧
    result = detector.process_frame(frame)
    assert result is not None, "应返回处理后的帧"
    assert detector.frame_count == 1, "帧计数应为 1"
    print(f"[OK] 单帧处理成功")
    
    # 测试批量处理
    frames = [frame, frame, frame]
    results = detector.process_batch(frames)
    assert len(results) == 3, "应返回 3 帧"
    assert detector.frame_count == 4, "帧计数应为 4"
    print(f"[OK] 批量处理成功")
    
    # 测试主导情绪
    detector.emotion_stats = {"Happy": 10, "Sad": 5, "Angry": 3}
    dominant, pct = detector._get_dominant_emotion()
    assert dominant == "Happy", "主导情绪应为 Happy"
    assert abs(pct - 55.6) < 0.1, f"Happy 应占约 55.6%，实际 {pct:.1f}%"
    print(f"[OK] 主导情绪: {dominant} ({pct:.1f}%)")
    
    # 测试情绪稳定度
    detector.history_recorder.history = [
        {"Happy": 60, "Sad": 20},
        {"Happy": 55, "Sad": 25},
        {"Happy": 65, "Sad": 15},
    ]
    stability = detector._get_emotion_stability()
    assert stability == 100.0, "稳定度应为 100%"
    print(f"[OK] 情绪稳定度: {stability:.0f}%")
    
    print("[PASS] 检测器集成测试通过")


def test_chart_generation():
    """测试 6: 图表生成"""
    print("\n=== 测试 6: 图表生成 ===")
    
    # 创建历史记录器并添加数据
    recorder = EmotionHistoryRecorder()
    for i in range(10):
        recorder.history.append({
            'timestamp': f'2026-11-11 14:{i*5:02d}:00',
            'Happy': 40 + i * 2,
            'Sad': 20 - i,
            'Angry': 10,
            'Fearful': 15,
            'Neutral': 25,
        })
    
    # 生成趋势图
    chart_path = "test_trend.png"
    success = recorder.generate_trend_chart(chart_path)
    assert success, "趋势图应生成成功"
    assert os.path.exists(chart_path), "趋势图文件应存在"
    print(f"[OK] 趋势图已生成: {chart_path}")
    
    # 清理
    os.remove(chart_path)
    
    print("[PASS] 图表生成测试通过")


if __name__ == "__main__":
    import numpy as np
    
    print("="*60)
    print("方案B 功能增强版 - 测试套件")
    print("="*60)
    
    try:
        test_config_loading()
        test_history_recorder()
        test_emotion_logger()
        test_video_recorder()
        test_detector_integration()
        test_chart_generation()
        
        print("\n" + "="*60)
        print("[OK] 所有测试通过!")
        print("="*60)
        
    except Exception as e:
        print(f"\n[FAIL] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
