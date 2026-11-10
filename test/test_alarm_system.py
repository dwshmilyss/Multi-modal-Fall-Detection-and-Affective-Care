"""测试智能告警系统核心逻辑"""
# -*- coding: utf-8 -*-
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from datetime import datetime, timedelta

# ==================== 告警管理器（复制核心代码用于测试）====================

class AlarmManager:
    """智能告警管理器"""
    
    WARNING = "warning"
    CRITICAL = "critical"
    
    HEART_RATE = "心率异常"
    SPO2 = "血氧异常"
    BLOOD_PRESSURE = "血压异常"
    TEMPERATURE = "体温异常"
    BATTERY = "电量不足"
    
    def __init__(self):
        self.alarms = []
        self.max_alarms = 500
        self.is_muted = False
        self.alarm_count = {self.WARNING: 0, self.CRITICAL: 0}
        self.alarm_type_count = {}
        self.last_alarm_time = None
        self.alarm_cooldown = 5
        self._last_alarm_by_type = {}
    
    def check_and_record(self, metric, value, status_text, status_color):
        if status_color == "#10B981":
            return None
        
        level = self.CRITICAL if status_color == "#EF4444" else self.WARNING
        
        alarm_type_map = {
            "heart_rate": self.HEART_RATE,
            "spo2": self.SPO2,
            "bp": self.BLOOD_PRESSURE,
            "temperature": self.TEMPERATURE,
            "battery": self.BATTERY
        }
        alarm_type = alarm_type_map.get(metric, "未知异常")
        
        now = datetime.now()
        last_time = self._last_alarm_by_type.get(alarm_type)
        if last_time and (now - last_time).total_seconds() < self.alarm_cooldown:
            return None
        
        alarm_record = {
            "time": now,
            "metric": metric,
            "alarm_type": alarm_type,
            "level": level,
            "value": value,
            "status": status_text,
            "color": status_color
        }
        
        self.alarms.append(alarm_record)
        if len(self.alarms) > self.max_alarms:
            self.alarms.pop(0)
        
        self.alarm_count[level] += 1
        self.alarm_type_count[alarm_type] = self.alarm_type_count.get(alarm_type, 0) + 1
        self.last_alarm_time = now
        self._last_alarm_by_type[alarm_type] = now
        
        return alarm_record
    
    def get_recent_alarms(self, count=20):
        return self.alarms[-count:] if self.alarms else []
    
    def get_alarm_stats(self):
        return {
            "total": len(self.alarms),
            "warning": self.alarm_count[self.WARNING],
            "critical": self.alarm_count[self.CRITICAL],
            "by_type": dict(self.alarm_type_count),
            "last_alarm": self.last_alarm_time
        }
    
    def clear_alarms(self):
        self.alarms.clear()
        self.alarm_count = {self.WARNING: 0, self.CRITICAL: 0}
        self.alarm_type_count.clear()
        self.last_alarm_time = None
        self._last_alarm_by_type.clear()
    
    def toggle_mute(self):
        self.is_muted = not self.is_muted
        return self.is_muted


# ==================== 测试代码 ====================

def test_alarm_manager():
    """测试告警管理器"""
    print("=" * 50)
    print("测试 AlarmManager")
    print("=" * 50)
    
    manager = AlarmManager()
    
    # 测试正常数据（不应产生告警）
    result = manager.check_and_record("heart_rate", 75, "正常平稳", "#10B981")
    assert result is None, "正常数据不应产生告警"
    print("✓ 正常数据不产生告警")
    
    # 测试异常数据（应产生告警）
    result = manager.check_and_record("heart_rate", 120, "心率过高", "#EF4444")
    assert result is not None, "异常数据应产生告警"
    assert result["level"] == AlarmManager.CRITICAL
    assert result["alarm_type"] == "心率异常"
    print(f"✓ 异常数据产生告警：{result['alarm_type']} - {result['level']}")
    
    # 测试冷却机制
    result2 = manager.check_and_record("heart_rate", 125, "心率过高", "#EF4444")
    assert result2 is None, "冷却期内不应重复告警"
    print("✓ 冷却机制正常工作")
    
    # 测试统计
    stats = manager.get_alarm_stats()
    assert stats["total"] == 1
    assert stats["critical"] == 1
    print(f"✓ 统计正确：总计={stats['total']}, 危险={stats['critical']}")
    
    # 测试静音切换
    is_muted = manager.toggle_mute()
    assert is_muted == True
    is_muted = manager.toggle_mute()
    assert is_muted == False
    print("✓ 静音切换正常")
    
    # 测试清空
    manager.clear_alarms()
    stats = manager.get_alarm_stats()
    assert stats["total"] == 0
    print("✓ 清空功能正常")
    
    print("\n✅ AlarmManager 测试通过！\n")


def test_alarm_levels():
    """测试告警级别"""
    print("=" * 50)
    print("测试告警级别")
    print("=" * 50)
    
    manager = AlarmManager()
    
    # 测试警告级别（黄色）
    result = manager.check_and_record("spo2", 93, "血氧偏低", "#F59E0B")
    assert result["level"] == AlarmManager.WARNING
    print(f"✓ 警告级别：{result['level']} - {result['alarm_type']}")
    
    # 测试危险级别（红色）
    result = manager.check_and_record("temperature", 38.5, "发热", "#EF4444")
    assert result["level"] == AlarmManager.CRITICAL
    print(f"✓ 危险级别：{result['level']} - {result['alarm_type']}")
    
    print("\n✅ 告警级别测试通过！\n")


def test_alarm_types():
    """测试不同告警类型"""
    print("=" * 50)
    print("测试告警类型")
    print("=" * 50)
    
    manager = AlarmManager()
    
    test_cases = [
        ("heart_rate", 110, "心率过高", "#EF4444", "心率异常"),
        ("spo2", 92, "血氧偏低", "#F59E0B", "血氧异常"),
        ("bp", (150, 95), "血压过高", "#EF4444", "血压异常"),
        ("temperature", 38.0, "发热", "#EF4444", "体温异常"),
        ("battery", 15, "电量不足", "#EF4444", "电量不足"),
    ]
    
    for metric, value, status, color, expected_type in test_cases:
        result = manager.check_and_record(metric, value, status, color)
        assert result is not None, f"{metric} 应产生告警"
        assert result["alarm_type"] == expected_type, f"期望 {expected_type}，实际 {result['alarm_type']}"
        print(f"✓ {metric}: {expected_type}")
    
    stats = manager.get_alarm_stats()
    assert stats["total"] == 5
    print(f"✓ 总计 {stats['total']} 条告警")
    
    print("\n✅ 告警类型测试通过！\n")


def test_cooldown_mechanism():
    """测试冷却机制"""
    print("=" * 50)
    print("测试冷却机制")
    print("=" * 50)
    
    manager = AlarmManager()
    manager.alarm_cooldown = 2  # 设置 2 秒冷却
    
    # 第一次告警
    result1 = manager.check_and_record("heart_rate", 120, "心率过高", "#EF4444")
    assert result1 is not None
    print("✓ 第一次告警成功")
    
    # 立即再次告警（应被冷却）
    result2 = manager.check_and_record("heart_rate", 125, "心率过高", "#EF4444")
    assert result2 is None
    print("✓ 冷却期内阻止重复告警")
    
    # 不同类型不受影响
    result3 = manager.check_and_record("spo2", 90, "血氧过低", "#EF4444")
    assert result3 is not None
    print("✓ 不同类型告警不受冷却影响")
    
    print("\n✅ 冷却机制测试通过！\n")


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("智能告警系统测试")
    print("=" * 50 + "\n")
    
    test_alarm_manager()
    test_alarm_levels()
    test_alarm_types()
    test_cooldown_mechanism()
    
    print("=" * 50)
    print("🎉 所有测试通过！")
    print("=" * 50)
