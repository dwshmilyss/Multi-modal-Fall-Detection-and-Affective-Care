from ultralytics import YOLO
import cv2
import sys
import os

def find_available_camera(max_index=5):
    """尝试找到可用的摄像头"""
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"✅ 找到可用摄像头：索引 {i}")
            return i
        cap.release()
    return None

def main():
    # 支持命令行参数：摄像头索引或视频文件路径
    # 用法：python app.py [摄像头索引|视频文件路径]
    # 示例：
    #   python app.py          # 自动检测摄像头
    #   python app.py 0        # 使用摄像头 0
    #   python app.py 1        # 使用摄像头 1
    #   python app.py video.mp4  # 使用视频文件
    
    source = 0  # 默认使用摄像头 0
    
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg.isdigit():
            source = int(arg)
        elif os.path.exists(arg):
            source = arg
        else:
            print(f"❌ 无效的参数：{arg}")
            print("用法：python app.py [摄像头索引|视频文件路径]")
            sys.exit(1)
    
    # 加载模型
    model_path = 'best.onnx'
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在：{model_path}")
        print("请确保 best.onnx 文件在当前目录下")
        sys.exit(1)
    
    print(f"📦 正在加载模型：{model_path}")
    model = YOLO(model_path)
    
    # 打开视频源
    if isinstance(source, int):
        # 摄像头模式
        print(f"📷 尝试打开摄像头 {source}...")
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print("❌ 无法打开摄像头")
            print("\n可能的原因：")
            print("  1. 电脑没有摄像头")
            print("  2. 摄像头被其他程序占用")
            print("  3. 摄像头驱动未安装")
            print("\n解决方案：")
            print("  - 使用视频文件测试：python app.py video.mp4")
            print("  - 尝试其他摄像头索引：python app.py 1")
            print("  - 检查摄像头是否被占用")
            
            # 尝试自动查找可用摄像头
            print("\n🔍 正在自动查找可用摄像头...")
            available_cam = find_available_camera()
            if available_cam is not None:
                print(f"✅ 找到可用摄像头：索引 {available_cam}")
                print(f"请使用命令：python app.py {available_cam}")
            else:
                print("❌ 未找到任何可用摄像头")
                print("建议使用视频文件进行测试")
            sys.exit(1)
        else:
            print(f"✅ 成功打开摄像头 {source}")
    else:
        # 视频文件模式
        print(f"🎬 打开视频文件：{source}")
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"❌ 无法打开视频文件：{source}")
            sys.exit(1)
    
    print("🚀 开始情绪检测，按 ESC 键退出...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            if isinstance(source, str):
                # 视频文件播放完毕，从头开始
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                print("❌ 无法读取帧")
                break  

        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_image_3d = cv2.merge([gray_image, gray_image, gray_image]) 
        
        results = model(gray_image_3d)
        result = results[0]

        try:
            annotated_frame = result.plot()
        except AttributeError:
            print("❌ plot() 方法不可用")
            break
        
        cv2.imshow('YOLO Inference', annotated_frame)
        
        if cv2.waitKey(1) == 27:  # ESC 键退出
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ 程序已退出")

if __name__ == "__main__":
    main()
