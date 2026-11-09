"""
生成跌倒测试视频
- 场景1: 人站立 -> 向前跌倒
- 场景2: 人站立 -> 向侧面跌倒  
- 场景3: 人从站立缓慢滑倒
用于测试跌倒检测系统的视频输入
"""
import numpy as np
import cv2
import math
import os

# 视频参数
WIDTH, HEIGHT = 640, 480
FPS = 30
OUTPUT_DIR = "test_videos"

def draw_person(frame, center_x, center_y, angle_deg=0, scale=1.0, color=(0, 255, 0)):
    """
    绘制一个简化的人体图形
    angle_deg: 身体倾斜角度 (0=站立, 90=水平)
    scale: 人体大小缩放
    """
    angle_rad = math.radians(angle_deg)
    s = scale
    
    # 身体各部分长度
    head_r = int(20 * s)
    torso_len = int(80 * s)
    arm_len = int(60 * s)
    leg_len = int(70 * s)
    
    # 头部位置 (身体顶部)
    head_y = center_y - int(torso_len * math.cos(angle_rad))
    head_x = center_x + int(torso_len * math.sin(angle_rad))
    
    # 画头 (圆形)
    cv2.circle(frame, (int(head_x), int(head_y)), head_r, color, -1)
    cv2.circle(frame, (int(head_x), int(head_y)), head_r, (255, 255, 255), 2)
    
    # 画身体躯干 (粗线)
    body_bottom = (center_x, center_y)
    cv2.line(frame, (int(head_x), int(head_y)), body_bottom, color, int(8 * s))
    
    # 画手臂 - 从肩膀位置
    shoulder_y = center_y - int(torso_len * 0.7 * math.cos(angle_rad))
    shoulder_x = center_x + int(torso_len * 0.7 * math.sin(angle_rad))
    
    # 左手臂 - 自然下垂或随跌倒姿势变化
    arm_angle = angle_rad + math.radians(30) if angle_deg < 45 else angle_rad - math.radians(60)
    left_hand_x = shoulder_x - int(arm_len * math.cos(arm_angle))
    left_hand_y = shoulder_y + int(arm_len * math.sin(arm_angle))
    cv2.line(frame, (int(shoulder_x), int(shoulder_y)), 
             (int(left_hand_x), int(left_hand_y)), color, int(6 * s))
    
    # 右手臂
    arm_angle2 = angle_rad - math.radians(30) if angle_deg < 45 else angle_rad + math.radians(60)
    right_hand_x = shoulder_x + int(arm_len * math.cos(arm_angle2))
    right_hand_y = shoulder_y + int(arm_len * math.sin(arm_angle2))
    cv2.line(frame, (int(shoulder_x), int(shoulder_y)),
             (int(right_hand_x), int(right_hand_y)), color, int(6 * s))
    
    # 画腿 - 从臀部(身体底部)
    # 左腿
    leg_angle = angle_rad + math.radians(15)
    left_foot_x = center_x - int(leg_len * math.sin(leg_angle))
    left_foot_y = center_y + int(leg_len * math.cos(leg_angle))
    cv2.line(frame, body_bottom, (int(left_foot_x), int(left_foot_y)), color, int(7 * s))
    
    # 右腿
    leg_angle2 = angle_rad - math.radians(15)
    right_foot_x = center_x + int(leg_len * math.sin(leg_angle2))
    right_foot_y = center_y + int(leg_len * math.cos(leg_angle2))
    cv2.line(frame, body_bottom, (int(right_foot_x), int(right_foot_y)), color, int(7 * s))
    
    return frame


def generate_fall_forward_video():
    """生成向前跌倒的视频"""
    output_path = os.path.join(OUTPUT_DIR, "fall_forward.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, FPS, (WIDTH, HEIGHT))
    
    total_frames = FPS * 5  # 5秒视频
    
    for i in range(total_frames):
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        
        # 添加地面
        cv2.rectangle(frame, (0, HEIGHT - 50), (WIDTH, HEIGHT), (50, 50, 50), -1)
        cv2.line(frame, (0, HEIGHT - 50), (WIDTH, HEIGHT - 50), (100, 100, 100), 2)
        
        # 计算跌倒动画
        t = i / total_frames  # 0 到 1 的进度
        
        if t < 0.3:
            # 站立阶段 (0-30%)
            person_x = WIDTH // 2
            person_y = HEIGHT - 180
            angle = 0
            color = (0, 255, 0)  # 绿色 = 正常
        elif t < 0.7:
            # 跌倒过程 (30%-70%)
            fall_progress = (t - 0.3) / 0.4
            # 使用缓动函数让动作更自然
            ease_progress = fall_progress ** 1.5
            person_x = WIDTH // 2 + int(50 * ease_progress)
            person_y = HEIGHT - 180 + int(100 * ease_progress)
            angle = int(90 * ease_progress)
            color = (0, 165, 255)  # 橙色 = 警告
        else:
            # 已跌倒 (70%-100%)
            person_x = WIDTH // 2 + 50
            person_y = HEIGHT - 80
            angle = 90
            color = (0, 0, 255)  # 红色 = 跌倒
        
        draw_person(frame, person_x, person_y, angle, 1.0, color)
        
        # 添加状态文字
        status_text = "STANDING" if t < 0.3 else "FALLING" if t < 0.7 else "FALLEN!"
        status_color = (0, 255, 0) if t < 0.3 else (0, 165, 255) if t < 0.7 else (0, 0, 255)
        cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, status_color, 2)
        
        # 添加帧信息
        cv2.putText(frame, f"Frame: {i}/{total_frames}", (20, HEIGHT - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        out.write(frame)
    
    out.release()
    print(f"[OK] 向前跌倒视频已生成: {output_path}")
    return output_path


def generate_fall_sideways_video():
    """生成向侧面跌倒的视频"""
    output_path = os.path.join(OUTPUT_DIR, "fall_sideways.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, FPS, (WIDTH, HEIGHT))
    
    total_frames = FPS * 4  # 4秒视频
    
    for i in range(total_frames):
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        
        # 添加地面
        cv2.rectangle(frame, (0, HEIGHT - 50), (WIDTH, HEIGHT), (50, 50, 50), -1)
        cv2.line(frame, (0, HEIGHT - 50), (WIDTH, HEIGHT - 50), (100, 100, 100), 2)
        
        t = i / total_frames
        
        if t < 0.25:
            # 站立
            person_x = WIDTH // 2
            person_y = HEIGHT - 180
            angle = 0
            color = (0, 255, 0)
        elif t < 0.65:
            # 向右侧跌倒
            fall_progress = (t - 0.25) / 0.4
            ease_progress = fall_progress ** 1.3
            person_x = WIDTH // 2 + int(80 * ease_progress)
            person_y = HEIGHT - 180 + int(80 * ease_progress)
            angle = int(85 * ease_progress)
            color = (0, 165, 255)
        else:
            # 已跌倒在地
            person_x = WIDTH // 2 + 80
            person_y = HEIGHT - 100
            angle = 85
            color = (0, 0, 255)
        
        draw_person(frame, person_x, person_y, angle, 1.0, color)
        
        status_text = "STANDING" if t < 0.25 else "FALLING" if t < 0.65 else "FALLEN!"
        status_color = (0, 255, 0) if t < 0.25 else (0, 165, 255) if t < 0.65 else (0, 0, 255)
        cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, status_color, 2)
        cv2.putText(frame, f"Frame: {i}/{total_frames}", (20, HEIGHT - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        out.write(frame)
    
    out.release()
    print(f"[OK] 侧面跌倒视频已生成: {output_path}")
    return output_path


def generate_normal_activity_video():
    """生成正常活动视频（作为对照组，不含跌倒）"""
    output_path = os.path.join(OUTPUT_DIR, "normal_activity.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, FPS, (WIDTH, HEIGHT))
    
    total_frames = FPS * 5  # 5秒视频
    
    for i in range(total_frames):
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        
        # 添加地面
        cv2.rectangle(frame, (0, HEIGHT - 50), (WIDTH, HEIGHT), (50, 50, 50), -1)
        cv2.line(frame, (0, HEIGHT - 50), (WIDTH, HEIGHT - 50), (100, 100, 100), 2)
        
        t = i / total_frames
        
        # 模拟正常行走 - 左右轻微摆动
        person_x = int(WIDTH * 0.3 + WIDTH * 0.4 * t)
        person_y = HEIGHT - 180 + int(5 * math.sin(t * 10))  # 轻微上下波动
        angle = int(5 * math.sin(t * 8))  # 轻微摇摆
        color = (0, 255, 0)
        
        draw_person(frame, person_x, person_y, angle, 1.0, color)
        
        cv2.putText(frame, "NORMAL WALKING", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2)
        cv2.putText(frame, f"Frame: {i}/{total_frames}", (20, HEIGHT - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        out.write(frame)
    
    out.release()
    print(f"[OK] 正常活动视频已生成: {output_path}")
    return output_path


def generate_combined_test_video():
    """生成包含多个场景的综合测试视频"""
    output_path = os.path.join(OUTPUT_DIR, "combined_test.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, FPS, (WIDTH, HEIGHT))
    
    # 场景1: 站立2秒 -> 跌倒2秒 -> 倒地1秒
    # 场景2: 正常行走2秒
    # 场景3: 再次跌倒2秒
    
    scenes = [
        ("stand", FPS * 2),      # 站立
        ("fall_forward", FPS * 2), # 向前跌倒
        ("fallen", FPS * 1),      # 倒地
        ("walk", FPS * 2),        # 行走
        ("fall_again", FPS * 2),  # 再次跌倒
    ]
    
    frame_idx = 0
    for scene_type, scene_frames in scenes:
        for j in range(scene_frames):
            frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
            
            # 地面
            cv2.rectangle(frame, (0, HEIGHT - 50), (WIDTH, HEIGHT), (50, 50, 50), -1)
            cv2.line(frame, (0, HEIGHT - 50), (WIDTH, HEIGHT - 50), (100, 100, 100), 2)
            
            if scene_type == "stand":
                person_x, person_y, angle = WIDTH // 2, HEIGHT - 180, 0
                color = (0, 255, 0)
                label = "Scene 1: Standing"
            elif scene_type == "fall_forward":
                progress = j / scene_frames
                person_x = WIDTH // 2 + int(60 * progress)
                person_y = HEIGHT - 180 + int(100 * progress)
                angle = int(90 * progress)
                color = (0, 165, 255)
                label = "Scene 2: Falling!"
            elif scene_type == "fallen":
                person_x, person_y, angle = WIDTH // 2 + 60, HEIGHT - 80, 90
                color = (0, 0, 255)
                label = "Scene 3: Fallen!"
            elif scene_type == "walk":
                progress = j / scene_frames
                person_x = int(100 + 200 * progress)
                person_y = HEIGHT - 180 + int(5 * math.sin(progress * 15))
                angle = int(5 * math.sin(progress * 10))
                color = (0, 255, 0)
                label = "Scene 4: Walking"
            elif scene_type == "fall_again":
                progress = j / scene_frames
                person_x = int(350 + 80 * progress)
                person_y = HEIGHT - 180 + int(100 * progress)
                angle = int(85 * progress)
                color = (0, 0, 255) if progress > 0.7 else (0, 165, 255)
                label = "Scene 5: Falling again!"
            
            draw_person(frame, person_x, person_y, angle, 1.0, color)
            cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.8, color, 2)
            cv2.putText(frame, f"Frame: {frame_idx}", (20, HEIGHT - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            out.write(frame)
            frame_idx += 1
    
    out.release()
    print(f"[OK] 综合测试视频已生成: {output_path}")
    return output_path


if __name__ == "__main__":
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 50)
    print("  生成跌倒检测测试视频")
    print("=" * 50)
    
    # 生成所有测试视频
    generate_fall_forward_video()
    generate_fall_sideways_video()
    generate_normal_activity_video()
    generate_combined_test_video()
    
    print("\n" + "=" * 50)
    print(f"  所有视频已保存到 {OUTPUT_DIR}/ 目录")
    print("=" * 50)
    print("\n生成的视频:")
    print("  1. fall_forward.mp4   - 向前跌倒")
    print("  2. fall_sideways.mp4  - 侧面跌倒")
    print("  3. normal_activity.mp4 - 正常行走(对照组)")
    print("  4. combined_test.mp4  - 综合场景")
    print("\n提示: 这些是简化图形动画，用于测试视频处理流程。")
    print("如需更真实的测试，建议使用实际跌倒视频数据集。")
