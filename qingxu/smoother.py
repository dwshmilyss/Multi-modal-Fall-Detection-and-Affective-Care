"""情绪时序平滑器 - 减少情绪跳动"""
from collections import deque


class EmotionSmoother:
    """情绪时序平滑器
    
    实现策略:
    1. 滑动窗口: 保存最近 N 帧的检测结果
    2. 切换稳定性: 最近连续 min_stable_frames 帧都是同一个新情绪才切换
    3. 置信度输出: 输出当前稳定情绪在窗口中的平均置信度
    """
    
    def __init__(self, window_size=10, min_stable_frames=3):
        """
        Args:
            window_size: 滑动窗口大小（帧数）
            min_stable_frames: 新情绪需要连续出现的最小帧数才切换
        """
        self.window_size = window_size
        self.min_stable_frames = min_stable_frames
        
        # 滑动窗口: 存储 (emotion, confidence) 元组
        self.window = deque(maxlen=window_size)
        
        # 当前稳定输出的情绪
        self.current_stable_emotion = None
        self.current_stable_confidence = 0.0
        
    def update(self, emotion, confidence):
        """更新一帧的检测结果
        
        切换策略: 基于最近连续帧判断
        - 如果最近 min_stable_frames 帧都是同一个新情绪，则切换
        - 否则保持当前稳定情绪
        
        Args:
            emotion: 检测到的情绪（英文）
            confidence: 置信度 0-1
            
        Returns:
            tuple: (稳定情绪, 稳定置信度)
        """
        if not emotion:
            return self.current_stable_emotion, self.current_stable_confidence
            
        # 添加到滑动窗口
        self.window.append((emotion, confidence))
        
        # 首次检测，直接输出
        if self.current_stable_emotion is None:
            self.current_stable_emotion = emotion
            self.current_stable_confidence = confidence
            return self.current_stable_emotion, self.current_stable_confidence
        
        # 检查最近 min_stable_frames 帧是否都是同一个情绪
        if len(self.window) >= self.min_stable_frames:
            recent = list(self.window)[-self.min_stable_frames:]
            recent_emotions = [emo for emo, _ in recent]
            
            # 如果最近 N 帧都是同一个情绪
            if len(set(recent_emotions)) == 1:
                new_emotion = recent_emotions[0]
                # 且与当前稳定情绪不同，则切换
                if new_emotion != self.current_stable_emotion:
                    self.current_stable_emotion = new_emotion
        
        # 计算当前稳定情绪的置信度（取窗口中该情绪的平均置信度）
        stable_confs = [conf for emo, conf in self.window if emo == self.current_stable_emotion]
        if stable_confs:
            self.current_stable_confidence = sum(stable_confs) / len(stable_confs)
        
        return self.current_stable_emotion, self.current_stable_confidence
    
    def reset(self):
        """重置平滑器状态"""
        self.window.clear()
        self.current_stable_emotion = None
        self.current_stable_confidence = 0.0
