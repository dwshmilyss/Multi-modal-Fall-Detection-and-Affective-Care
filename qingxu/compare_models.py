"""
模型对比测试脚本
对比 YOLO11n/s/m 模型的检测效果

使用方法:
    python compare_models.py

按数字键切换模型，观察同一场景下的检测结果
"""
import cv2
import numpy as np
from ultralytics import YOLO
import time
from pathlib import Path

def compare_models():
    """对比不同模型的检测效果"""
    
    # 模型列表
    models_config = [
        {"path": "best.onnx", "name": "YOLO11n (原始)"},
        {"path": "best_yolo11s.onnx", "name": "YOLO11s (新)"},
        {"path": "best_yolo11m.onnx", "name": "YOLO11m (新)"},
    ]
    
    # 加载存在的模型
    models = []
    for cfg in models_config:
        model_path = Path(__file__).parent / cfg["path"]
        if model_path.exists():
            print(f"加载模型: {cfg['name']} ({cfg['path']})")
            models.append({
                "model": YOLO(str(model_path)),
                "name": cfg["name"],
                "path": cfg["path"],
            })
        else:
            print(f"跳过（文件不存在）: {cfg['path']}")
    
    if not models:
        print("\n错误: 没有找到任何模型文件")
        print("请先运行训练脚本或确保模型文件存在")
        return
    
    # 打开摄像头
    cap = None
    backends = [
        (cv2.CAP_MSMF, "MSMF"),
        (cv2.CAP_DSHOW, "DSHOW"),
        (cv2.CAP_ANY, "ANY"),
    ]
    
    for backend, name in backends:
        for index in range(3):
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    print(f"\n✓ 成功打开摄像头 {index} (后端: {name})")
                    break
            cap.release()
        else:
            continue
        break
    
    if not cap or not cap.isOpened():
        print("\n错误: 无法打开摄像头")
        print("请检查摄像头连接和权限设置")
        return
    
    print("\n" + "="*60)
    print("模型对比测试")
    print("="*60)
    print("\n可用模型:")
    for i, m in enumerate(models, 1):
        print(f"  [{i}] {m['name']}")
    print("\n操作说明:")
    print("  按数字键 1-3 切换模型")
    print("  按 ESC 退出")
    print("="*60)
    
    current_idx = 0
    frame_count = 0
    emotion_counts = {}
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        m = models[current_idx]
        
        # 推理
        results = m["model"](frame, conf=0.3, imgsz=640, verbose=False)
        result = results[0]
        
        # 统计情绪
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                emotion = result.names.get(cls_id, 'Unknown')
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        # 绘制结果
        annotated = result.plot()
        
        # 显示信息
        h, w = annotated.shape[:2]
        
        # 当前模型
        cv2.putText(annotated, f"[{current_idx+1}] {m['name']}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # FPS
        frame_count += 1
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        cv2.putText(annotated, f"FPS: {fps:.1f}", (w-150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 情绪统计
        y_offset = 100
        cv2.putText(annotated, "Emotion Stats:", (w-200, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 25
        for emotion, count in sorted(emotion_counts.items(), key=lambda x: -x[1]):
            text = f"{emotion}: {count}"
            color = (0, 255, 0) if emotion == 'Angry' else (200, 200, 200)
            cv2.putText(annotated, text, (w-200, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += 20
        
        # 提示
        cv2.putText(annotated, "Press 1-3 switch model, ESC exit",
                    (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.imshow('Model Comparison', annotated)
        
        # 处理按键
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key in [ord('1'), ord('2'), ord('3')]:
            new_idx = key - ord('1')
            if new_idx < len(models):
                current_idx = new_idx
                frame_count = 0
                emotion_counts = {}
                start_time = time.time()
                print(f"\n切换到模型 [{current_idx+1}] {models[current_idx]['name']}")
        elif key == ord('r'):  # 重置统计
            frame_count = 0
            emotion_counts = {}
            start_time = time.time()
            print("\n统计已重置")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "="*60)
    print("对比测试完成")
    print("="*60)

if __name__ == '__main__':
    compare_models()
