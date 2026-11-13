# themes.py — 6套UI主题注册中心 + QSS生成器
# 所有主题使用同一套 key 结构，apply_theme() 无需判断主题类型

import ctypes
import ctypes.wintypes
import sys
import os

# ==================== Windows DWM API 真毛玻璃 ====================

# DWM 常量
DWMWA_SYSTEMBACKDROP_TYPE = 38  # Windows 11 22H2+
DWMWA_USE_IMMERSIVE_DARK_MODE = 20

# 系统背景类型
DWMSBT_MAINWINDOW = 1       # Mica
DWMSBT_TRANSIENTWINDOW = 2  # Acrylic（真毛玻璃）
DWMSBT_TABBEDWINDOW = 3     # Tabbed

def _get_dwm_dll():
    """加载 DWM API DLL"""
    try:
        return ctypes.windll.dwmapi
    except Exception:
        return None

def _get_user32_dll():
    """加载 User32 DLL"""
    try:
        return ctypes.windll.user32
    except Exception:
        return None

def apply_dwm_glass(hwnd, theme: dict):
    """
    对指定窗口句柄应用 Windows DWM 真毛玻璃效果
    
    Args:
        hwnd: 窗口句柄 (int)
        theme: 主题字典，需要包含 glass_opacity 字段
    
    Returns:
        bool: 是否成功应用
    """
    if sys.platform != "win32":
        return False
    
    # 只有毛玻璃主题才启用 DWM 效果
    if theme.get("glass_opacity", 0) <= 0:
        return False
    
    dwm = _get_dwm_dll()
    if not dwm:
        return False
    
    # 检查 Windows 版本（需要 Win11 22H2+ 才能用 SYSTEMBACKDROP_TYPE）
    try:
        import platform
        win_ver = platform.version().split(".")
        major = int(win_ver[0])
        build = int(win_ver[2]) if len(win_ver) > 2 else 0
        
        # Windows 11 22H2 (build 22621+) 支持 DWMWA_SYSTEMBACKDROP_TYPE
        if major >= 10 and build >= 22621:
            # 使用 Acrylic 真毛玻璃效果
            backdrop_type = ctypes.c_int(DWMSBT_TRANSIENTWINDOW)
            result = dwm.DwmSetWindowAttribute(
                ctypes.wintypes.HWND(hwnd),
                DWMWA_SYSTEMBACKDROP_TYPE,
                ctypes.byref(backdrop_type),
                ctypes.sizeof(backdrop_type)
            )
            return result == 0  # S_OK = 0
        else:
            # Windows 10 或早期 Win11：使用 DwmEnableBlurBehindWindow
            return _apply_blur_behind(hwnd)
    except Exception:
        return False

def _apply_blur_behind(hwnd):
    """Windows 10 回退方案：DwmEnableBlurBehindWindow"""
    dwm = _get_dwm_dll()
    if not dwm:
        return False
    
    try:
        class DWM_BLURBEHIND(ctypes.Structure):
            _fields_ = [
                ("dwFlags", ctypes.wintypes.DWORD),
                ("fEnable", ctypes.wintypes.BOOL),
                ("hRgnBlur", ctypes.wintypes.HANDLE),
                ("fTransitionOnMaximized", ctypes.wintypes.BOOL),
            ]
        
        blur = DWM_BLURBEHIND()
        blur.dwFlags = 1  # DWM_BB_ENABLE
        blur.fEnable = True
        blur.hRgnBlur = None
        blur.fTransitionOnMaximized = False
        
        result = dwm.DwmEnableBlurBehindWindow(
            ctypes.wintypes.HWND(hwnd),
            ctypes.byref(blur)
        )
        return result == 0
    except Exception:
        return False

def apply_dwm_dark_mode(hwnd, is_dark: bool = True):
    """应用 Windows 10/11 深色模式到窗口标题栏"""
    if sys.platform != "win32":
        return False
    
    dwm = _get_dwm_dll()
    if not dwm:
        return False
    
    try:
        value = ctypes.c_int(1 if is_dark else 0)
        result = dwm.DwmSetWindowAttribute(
            ctypes.wintypes.HWND(hwnd),
            DWMWA_USE_IMMERSIVE_DARK_MODE,
            ctypes.byref(value),
            ctypes.sizeof(value)
        )
        return result == 0
    except Exception:
        return False

THEME_REGISTRY = {
    "one_dark": {
        "name": "ONE DARK",
        "icon": "\U0001f311",
        # === 全局色板 ===
        "bg_main":        "#1E222A",
        "bg_sidebar":     "#21252B",
        "bg_card":        "#21252B",
        "bg_video":       "#181A1F",
        "bg_alarm":       "#181A1F",
        "bg_log":         "#181A1F",
        "bg_input":       "#21252B",
        "bg_btn":         "#21252B",
        "bg_btn_hover":   "#2C313C",
        "bg_btn_pressed": "#1E222A",
        "bg_disabled":    "#1E222A",
        # === 边框色 ===
        "border":         "#282C34",
        "border_light":   "#3E4452",
        "sidebar_border": "#181A1F",
        # === 文字色 ===
        "text_primary":   "#ABB2BF",
        "text_secondary": "#5C6370",
        "text_title":     "#61AFEF",
        "text_gold":      "#E5C07B",
        "text_clock":     "#98C379",
        "text_log":       "#98C379",
        "text_video_hint":"#528BFF",
        # === 功能色 ===
        "accent":         "#61AFEF",
        "success":        "#98C379",
        "success_hover":  "#88B369",
        "danger":         "#E06C75",
        "danger_hover":   "#D05C65",
        # === 下拉框专属 ===
        "combo_selection_bg": "#2C313C",
        "combo_selection_fg": "#61AFEF",
        # === 毛玻璃参数 ===
        "glass_opacity":  0.0,
        "glass_blur":     0,
        "glass_border_a": 0.0,
    },
    "warm_care": {
        "name": "暖白关怀风",
        "icon": "\U0001f33b",
        "bg_main":        "#FAF8F5",
        "bg_sidebar":     "#FFF5EB",
        "bg_card":        "#FFFFFF",
        "bg_video":       "#F5EDE5",
        "bg_alarm":       "#FFF0EB",
        "bg_log":         "#F8F2EC",
        "bg_input":       "#FFF8F2",
        "bg_btn":         "#FFFFFF",
        "bg_btn_hover":   "#F5E6D8",
        "bg_btn_pressed": "#EDE0D4",
        "bg_disabled":    "#F5F0EB",
        "border":         "#F0E0D0",
        "border_light":   "#E8D5C0",
        "sidebar_border": "#F0E0D0",
        "text_primary":   "#4A3728",
        "text_secondary": "#8B7355",
        "text_title":     "#D4874E",
        "text_gold":      "#C47A3E",
        "text_clock":     "#5EAD85",
        "text_log":       "#6B8F5E",
        "text_video_hint":"#D4874E",
        "accent":         "#FF8C69",
        "success":        "#7EC8B0",
        "success_hover":  "#6EB8A0",
        "danger":         "#E8837C",
        "danger_hover":   "#D8736C",
        "combo_selection_bg": "#F5E6D8",
        "combo_selection_fg": "#FF8C69",
        "glass_opacity":  0.0,
        "glass_blur":     0,
        "glass_border_a": 0.0,
    },
    "medical_blue": {
        "name": "医疗科技蓝",
        "icon": "\U0001f489",
        "bg_main":        "#0D1B2A",
        "bg_sidebar":     "#0A1628",
        "bg_card":        "#132D46",
        "bg_video":       "#081420",
        "bg_alarm":       "#0A1828",
        "bg_log":         "#081420",
        "bg_input":       "#0F2236",
        "bg_btn":         "#132D46",
        "bg_btn_hover":   "#1A3A5C",
        "bg_btn_pressed": "#0D1B2A",
        "bg_disabled":    "#0A1628",
        "border":         "#1A3A5C",
        "border_light":   "#254A6C",
        "sidebar_border": "#152840",
        "text_primary":   "#E8F4F8",
        "text_secondary": "#8BAAB8",
        "text_title":     "#4FC0D0",
        "text_gold":      "#4FC0D0",
        "text_clock":     "#48BB78",
        "text_log":       "#4FC0D0",
        "text_video_hint":"#4FC0D0",
        "accent":         "#4FC0D0",
        "success":        "#48BB78",
        "success_hover":  "#38AB68",
        "danger":         "#FC8181",
        "danger_hover":   "#EC7171",
        "combo_selection_bg": "#1A3A5C",
        "combo_selection_fg": "#4FC0D0",
        "glass_opacity":  0.0,
        "glass_blur":     0,
        "glass_border_a": 0.0,
    },
    "frosted_glass": {
        "name": "毛玻璃暗色",
        "icon": "\u2728",
        "bg_main":        "#1A1B2E",
        "bg_sidebar":     "rgba(255,255,255,0.04)",
        "bg_card":        "rgba(255,255,255,0.06)",
        "bg_video":       "rgba(0,0,0,0.30)",
        "bg_alarm":       "rgba(102,126,234,0.10)",
        "bg_log":         "rgba(0,0,0,0.25)",
        "bg_input":       "rgba(255,255,255,0.05)",
        "bg_btn":         "rgba(255,255,255,0.06)",
        "bg_btn_hover":   "rgba(255,255,255,0.10)",
        "bg_btn_pressed": "rgba(255,255,255,0.03)",
        "bg_disabled":    "rgba(255,255,255,0.02)",
        "border":         "rgba(255,255,255,0.10)",
        "border_light":   "rgba(255,255,255,0.15)",
        "sidebar_border": "rgba(255,255,255,0.08)",
        "text_primary":   "#E8E8F0",
        "text_secondary": "#9898B0",
        "text_title":     "#A78BFA",
        "text_gold":      "#A78BFA",
        "text_clock":     "#00F5D4",
        "text_log":       "#00F5D4",
        "text_video_hint":"#A78BFA",
        "accent":         "#667EEA",
        "success":        "#00F5D4",
        "success_hover":  "#00E5C4",
        "danger":         "#FF6B8A",
        "danger_hover":   "#EF5B7A",
        "combo_selection_bg": "rgba(255,255,255,0.10)",
        "combo_selection_fg": "#667EEA",
        "glass_opacity":  0.06,
        "glass_blur":     20,
        "glass_border_a": 0.10,
    },
    "nature_green": {
        "name": "自然健康绿",
        "icon": "\U0001f33f",
        "bg_main":        "#1A2E1A",
        "bg_sidebar":     "#152615",
        "bg_card":        "#243824",
        "bg_video":       "#122012",
        "bg_alarm":       "#1A301A",
        "bg_log":         "#122012",
        "bg_input":       "#1E321E",
        "bg_btn":         "#243824",
        "bg_btn_hover":   "#2A4A2A",
        "bg_btn_pressed": "#1A2E1A",
        "bg_disabled":    "#152615",
        "border":         "#2D5A2D",
        "border_light":   "#3A6A3A",
        "sidebar_border": "#2A4A2A",
        "text_primary":   "#F0EDE5",
        "text_secondary": "#A8B8A0",
        "text_title":     "#95D5B2",
        "text_gold":      "#95D5B2",
        "text_clock":     "#F4D35E",
        "text_log":       "#95D5B2",
        "text_video_hint":"#95D5B2",
        "accent":         "#95D5B2",
        "success":        "#95D5B2",
        "success_hover":  "#85C5A2",
        "danger":         "#E8837C",
        "danger_hover":   "#D8736C",
        "combo_selection_bg": "#2A4A2A",
        "combo_selection_fg": "#95D5B2",
        "glass_opacity":  0.0,
        "glass_blur":     0,
        "glass_border_a": 0.0,
    },
    "hybrid": {
        "name": "医疗蓝+毛玻璃",
        "icon": "\U0001f30a",
        "bg_main":        "#0D1B2A",
        "bg_sidebar":     "rgba(10,22,40,0.70)",
        "bg_card":        "rgba(79,192,208,0.08)",
        "bg_video":       "rgba(8,20,32,0.60)",
        "bg_alarm":       "rgba(79,192,208,0.06)",
        "bg_log":         "rgba(0,0,0,0.30)",
        "bg_input":       "rgba(10,22,40,0.50)",
        "bg_btn":         "rgba(79,192,208,0.08)",
        "bg_btn_hover":   "rgba(79,192,208,0.15)",
        "bg_btn_pressed": "rgba(79,192,208,0.04)",
        "bg_disabled":    "rgba(10,22,40,0.40)",
        "border":         "rgba(79,192,208,0.15)",
        "border_light":   "rgba(79,192,208,0.25)",
        "sidebar_border": "rgba(79,192,208,0.15)",
        "text_primary":   "#E8F4F8",
        "text_secondary": "#8BAAB8",
        "text_title":     "#4FC0D0",
        "text_gold":      "#4FC0D0",
        "text_clock":     "#667EEA",
        "text_log":       "#4FC0D0",
        "text_video_hint":"#4FC0D0",
        "accent":         "#4FC0D0",
        "success":        "#4FC0D0",
        "success_hover":  "#3FB0C0",
        "danger":         "#FF6B8A",
        "danger_hover":   "#EF5B7A",
        "combo_selection_bg": "rgba(79,192,208,0.15)",
        "combo_selection_fg": "#4FC0D0",
        "glass_opacity":  0.08,
        "glass_blur":     16,
        "glass_border_a": 0.15,
    },
}


def generate_global_stylesheet(t: dict) -> str:
    """根据主题字典生成全局 QSS 样式表"""
    # 弹出层强制使用不透明背景（bg_main 永远是纯色），避免毛玻璃主题下重影
    popup_bg = t['bg_main']
    popup_sel = t['combo_selection_bg'] if not t['combo_selection_bg'].startswith('rgba') else t['bg_btn_hover'] if not t['bg_btn_hover'].startswith('rgba') else t['border']
    return f"""
QMainWindow {{
    background-color: {t['bg_main']};
}}
QLabel {{
    font-family: 'Microsoft YaHei', 'Segoe UI';
    color: {t['text_primary']};
}}
QPushButton.sidebar_btn {{
    background-color: transparent;
    border: none;
    border-radius: 6px;
    color: {t['text_primary']};
    text-align: left;
    padding: 10px 14px;
    font-family: 'Microsoft YaHei';
    font-size: 14px;
    font-weight: 500;
}}
QPushButton.sidebar_btn:hover {{
    background-color: {t['bg_btn_hover']};
    color: {t['accent']};
}}
QPushButton.sidebar_btn:pressed {{
    background-color: {t['bg_btn_pressed']};
}}
QPushButton {{
    background-color: {t['bg_btn']};
    border: 1px solid {t['border_light']};
    border-radius: 6px;
    color: {t['text_primary']};
    padding: 8px 14px;
    font-family: 'Microsoft YaHei';
    font-size: 13px;
    font-weight: 600;
}}
QPushButton:hover {{
    background-color: {t['bg_btn_hover']};
    border-color: {t['accent']};
    color: {t['accent']};
}}
QPushButton:pressed {{
    background-color: {t['bg_btn_pressed']};
}}
QPushButton:disabled {{
    background-color: {t['bg_disabled']};
    color: {t['text_secondary']};
    border: 1px solid {t['border']};
}}
QComboBox {{
    background-color: {t['bg_input']};
    border: 1px solid {t['border_light']};
    border-radius: 6px;
    color: {t['text_primary']};
    font-size: 13px;
    font-weight: 600;
    padding: 6px 10px;
    font-family: 'Microsoft YaHei';
}}
QComboBox:hover {{
    border-color: {t['accent']};
}}
QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 28px;
    border-left-width: 1px;
    border-left-color: {t['border_light']};
    border-left-style: solid;
}}
QComboBox::down-arrow {{
    image: none;
    width: 0;
    height: 0;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid {t['text_primary']};
    margin-right: 8px;
}}
QComboBox QAbstractItemView {{
    background-color: {popup_bg};
    color: {t['text_primary']};
    selection-background-color: {popup_sel};
    selection-color: {t['combo_selection_fg']};
    border: 1px solid {t['border_light']};
    font-size: 13px;
    padding: 4px;
}}
"""


def apply_glass_effect(widget, t: dict):
    """对指定 widget 应用毛玻璃模拟效果（半透明+圆角+边框）"""
    if t["glass_opacity"] <= 0:
        return
    opacity = t["glass_opacity"]
    border_a = t["glass_border_a"]
    widget.setStyleSheet(f"""
        QFrame {{
            background-color: rgba(255,255,255,{opacity});
            border: 1px solid rgba(255,255,255,{border_a});
            border-radius: 10px;
        }}
    """)


def save_theme(theme_key: str):
    """保存当前主题到本地 JSON"""
    import json, os
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "theme_config.json")
    try:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump({"theme": theme_key}, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def load_theme(default: str = "one_dark") -> str:
    """从本地 JSON 加载主题，失败则返回默认值"""
    import json, os
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "theme_config.json")
    try:
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                key = data.get("theme", default)
                if key in THEME_REGISTRY:
                    return key
    except Exception:
        pass
    return default


def get_current_theme() -> dict:
    """获取当前主题字典"""
    key = load_theme("one_dark")
    return THEME_REGISTRY.get(key, THEME_REGISTRY["one_dark"])


def apply_theme_to_dialog(dialog, theme: dict = None):
    """
    对 QDialog 应用主题样式
    
    Args:
        dialog: QDialog 实例
        theme: 主题字典，默认使用当前主题
    """
    if theme is None:
        theme = get_current_theme()
    
    t = theme
    
    # 弹出层强制使用不透明背景，避免毛玻璃主题下重影
    popup_bg = t['bg_main']
    popup_sel = t['combo_selection_bg'] if not t['combo_selection_bg'].startswith('rgba') else (t['bg_btn_hover'] if not t['bg_btn_hover'].startswith('rgba') else t['border'])
    
    # 对话框背景
    dialog.setStyleSheet(f"""
        QDialog {{
            background-color: {t['bg_main']};
            color: {t['text_primary']};
            font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif;
        }}
        QLabel {{
            color: {t['text_primary']};
            border: none;
        }}
        QPushButton {{
            background-color: {t['bg_btn']};
            border: 1px solid {t['border_light']};
            border-radius: 6px;
            color: {t['text_primary']};
            font-size: 13px;
            font-weight: 600;
            padding: 8px 14px;
        }}
        QPushButton:hover {{
            background-color: {t['bg_btn_hover']};
            border-color: {t['accent']};
            color: {t['accent']};
        }}
        QPushButton:pressed {{
            background-color: {t['bg_btn_pressed']};
        }}
        QLineEdit {{
            background-color: {t['bg_input']};
            border: 1px solid {t['border_light']};
            border-radius: 6px;
            color: {t['text_primary']};
            padding: 6px 10px;
            font-size: 13px;
        }}
        QLineEdit:focus {{
            border-color: {t['accent']};
        }}
        QTextEdit {{
            background-color: {t['bg_input']};
            border: 1px solid {t['border_light']};
            border-radius: 6px;
            color: {t['text_primary']};
            font-size: 13px;
            padding: 6px 10px;
        }}
        QListWidget {{
            background-color: {t['bg_input']};
            border: 1px solid {t['border_light']};
            border-radius: 6px;
            color: {t['text_primary']};
            font-size: 13px;
        }}
        QListWidget::item {{
            padding: 6px;
            border-bottom: 1px solid {t['border']};
        }}
        QListWidget::item:selected {{
            background-color: {popup_sel};
            color: {t['combo_selection_fg']};
        }}
        QComboBox {{
            background-color: {t['bg_input']};
            border: 1px solid {t['border_light']};
            border-radius: 6px;
            color: {t['text_primary']};
            padding: 6px 10px;
            font-size: 13px;
        }}
        QComboBox QAbstractItemView {{
            background-color: {popup_bg};
            color: {t['text_primary']};
            selection-background-color: {popup_sel};
            selection-color: {t['combo_selection_fg']};
        }}
        QFrame {{
            background-color: {t['bg_card']};
            border: 1px solid {t['border']};
            border-radius: 8px;
        }}
        QSlider::groove:horizontal {{
            border: 1px solid {t['border_light']};
            height: 6px;
            background: {t['bg_input']};
            border-radius: 3px;
        }}
        QSlider::handle:horizontal {{
            background: {t['accent']};
            border: 1px solid {t['border_light']};
            width: 16px;
            margin: -5px 0;
            border-radius: 8px;
        }}
        QCheckBox {{
            color: {t['text_primary']};
            font-size: 13px;
        }}
        QCheckBox::indicator {{
            width: 18px;
            height: 18px;
            border-radius: 4px;
            border: 1px solid {t['border_light']};
            background-color: {t['bg_input']};
        }}
        QCheckBox::indicator:checked {{
            background-color: {t['accent']};
            border-color: {t['accent']};
        }}
    """)
    
    # 应用 DWM 毛玻璃效果
    if sys.platform == "win32":
        try:
            hwnd = int(dialog.winId())
            apply_dwm_glass(hwnd, t)
            is_dark = t.get("bg_main", "#000000").startswith("#") and t["bg_main"] < "#808080"
            apply_dwm_dark_mode(hwnd, is_dark)
        except Exception:
            pass


# ==================== QMessageBox 主题适配 ====================

def apply_theme_to_messagebox(msgbox, t: dict):
    """为 QMessageBox 应用当前主题样式"""
    msgbox.setStyleSheet(f"""
        QMessageBox {{
            background-color: {t['bg_main']};
            color: {t['text_primary']};
        }}
        QMessageBox QLabel {{
            color: {t['text_primary']};
            font-size: 13px;
            padding: 10px;
        }}
        QMessageBox QPushButton {{
            background-color: {t['bg_btn']};
            color: {t['text_primary']};
            border: 1px solid {t['border_light']};
            border-radius: 6px;
            padding: 8px 20px;
            font-size: 13px;
            min-width: 80px;
        }}
        QMessageBox QPushButton:hover {{
            background-color: {t['bg_btn_hover']};
            border-color: {t['accent']};
            color: {t['accent']};
        }}
        QMessageBox QPushButton:pressed {{
            background-color: {t['bg_btn_pressed']};
        }}
    """)
    # 应用 DWM 效果
    if sys.platform == "win32":
        try:
            hwnd = int(msgbox.winId())
            apply_dwm_glass(hwnd, t)
            is_dark = t.get("bg_main", "#000000").startswith("#") and t["bg_main"] < "#808080"
            apply_dwm_dark_mode(hwnd, is_dark)
        except Exception:
            pass


def themed_info(parent, title: str, text: str):
    """主题化的信息提示框"""
    from PyQt6.QtWidgets import QMessageBox
    t = get_current_theme()
    msg = QMessageBox(parent)
    msg.setIcon(QMessageBox.Icon.Information)
    msg.setWindowTitle(title)
    msg.setText(text)
    msg.setStandardButtons(QMessageBox.StandardButton.Ok)
    apply_theme_to_messagebox(msg, t)
    return msg.exec()


def themed_warning(parent, title: str, text: str):
    """主题化的警告提示框"""
    from PyQt6.QtWidgets import QMessageBox
    t = get_current_theme()
    msg = QMessageBox(parent)
    msg.setIcon(QMessageBox.Icon.Warning)
    msg.setWindowTitle(title)
    msg.setText(text)
    msg.setStandardButtons(QMessageBox.StandardButton.Ok)
    apply_theme_to_messagebox(msg, t)
    return msg.exec()


def themed_critical(parent, title: str, text: str):
    """主题化的错误提示框"""
    from PyQt6.QtWidgets import QMessageBox
    t = get_current_theme()
    msg = QMessageBox(parent)
    msg.setIcon(QMessageBox.Icon.Critical)
    msg.setWindowTitle(title)
    msg.setText(text)
    msg.setStandardButtons(QMessageBox.StandardButton.Ok)
    apply_theme_to_messagebox(msg, t)
    return msg.exec()


def themed_question(parent, title: str, text: str, 
                    buttons=QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    default=QMessageBox.StandardButton.No):
    """主题化的询问提示框"""
    from PyQt6.QtWidgets import QMessageBox
    t = get_current_theme()
    msg = QMessageBox(parent)
    msg.setIcon(QMessageBox.Icon.Question)
    msg.setWindowTitle(title)
    msg.setText(text)
    msg.setStandardButtons(buttons)
    msg.setDefaultButton(default)
    apply_theme_to_messagebox(msg, t)
    return msg.exec()
