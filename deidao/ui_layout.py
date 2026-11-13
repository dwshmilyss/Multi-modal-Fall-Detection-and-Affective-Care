# ui_layout.py — 支持6套主题实时切换的主界面布局
from PyQt6.QtWidgets import (QWidget, QLabel, QPushButton, QFrame,
                             QListWidget, QHBoxLayout, QVBoxLayout, QSizePolicy, QComboBox)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from themes import THEME_REGISTRY, generate_global_stylesheet, apply_glass_effect


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1280, 800)
        MainWindow.setWindowTitle("AI Vision Guardian - 智能人员跌倒实时监测系统")

        self.centralwidget = QWidget(MainWindow)

        # 主横向布局：左侧菜单栏 + 右侧主工作区
        horizontalLayout_main = QHBoxLayout(self.centralwidget)
        horizontalLayout_main.setSpacing(0)
        horizontalLayout_main.setContentsMargins(0, 0, 0, 0)

        # ==================== 1. 左侧侧边栏 (Sidebar) ====================
        self.sidebar_frame = QFrame(self.centralwidget)
        self.sidebar_frame.setFixedWidth(230)
        verticalLayout_sidebar = QVBoxLayout(self.sidebar_frame)
        verticalLayout_sidebar.setSpacing(8)
        verticalLayout_sidebar.setContentsMargins(12, 16, 12, 16)

        # 侧边栏标题
        self.sidebar_title = QLabel("ONE DARK", self.sidebar_frame)
        self.sidebar_title.setFont(QFont("Microsoft YaHei", 14, QFont.Weight.Bold))
        verticalLayout_sidebar.addWidget(self.sidebar_title)

        # 间隔线
        self.line_sidebar = QFrame(self.sidebar_frame)
        self.line_sidebar.setFrameShape(QFrame.Shape.HLine)
        self.line_sidebar.setFixedHeight(1)
        verticalLayout_sidebar.addWidget(self.line_sidebar)

        # ★ 主题切换下拉框（新增）
        self.theme_combo = QComboBox(self.sidebar_frame)
        for key, info in THEME_REGISTRY.items():
            self.theme_combo.addItem(f"{info['icon']} {info['name']}", key)
        verticalLayout_sidebar.addWidget(self.theme_combo)

        # 菜单按钮
        self.btn_menu_home = QPushButton("实时监控", self.sidebar_frame)
        self.btn_menu_home.setProperty("class", "sidebar_btn")
        verticalLayout_sidebar.addWidget(self.btn_menu_home)

        # 摄像头选择下拉框
        self.camera_combo = QComboBox(self.sidebar_frame)
        self.camera_combo.addItem("摄像头 0 (默认)")
        verticalLayout_sidebar.addWidget(self.camera_combo)
        self.btn_camera = self.camera_combo

        self.btn_file = QPushButton("导入视频", self.sidebar_frame)
        self.btn_file.setProperty("class", "sidebar_btn")
        verticalLayout_sidebar.addWidget(self.btn_file)

        self.btn_history = QPushButton("历史片段", self.sidebar_frame)
        self.btn_history.setProperty("class", "sidebar_btn")
        verticalLayout_sidebar.addWidget(self.btn_history)

        # 弹簧将底部按钮推到底端
        verticalLayout_sidebar.addStretch()

        # 底部两个按钮
        self.btn_settings = QPushButton("系统高级设置", self.sidebar_frame)
        self.btn_settings.setProperty("class", "sidebar_btn")
        verticalLayout_sidebar.addWidget(self.btn_settings)

        self.btn_account = QPushButton("切换账号", self.sidebar_frame)
        self.btn_account.setProperty("class", "sidebar_btn")
        verticalLayout_sidebar.addWidget(self.btn_account)

        horizontalLayout_main.addWidget(self.sidebar_frame)

        # ==================== 2. 右侧主内容区域 ====================
        self.right_container = QWidget(self.centralwidget)
        verticalLayout_right_container = QVBoxLayout(self.right_container)
        verticalLayout_right_container.setSpacing(12)
        verticalLayout_right_container.setContentsMargins(16, 16, 16, 16)

        # 顶部栏
        self.header_frame = QFrame(self.right_container)
        horizontalLayout_header = QHBoxLayout(self.header_frame)
        horizontalLayout_header.setContentsMargins(20, 12, 20, 12)

        self.title_label = QLabel("管理平台 - 智能人员跌倒实时监测系统", self.header_frame)
        self.title_label.setFont(QFont("Microsoft YaHei", 13, QFont.Weight.Bold))
        horizontalLayout_header.addWidget(self.title_label)

        horizontalLayout_header.addStretch()

        self.lbl_clock = QLabel("--:--:--", self.header_frame)
        self.lbl_clock.setFont(QFont("Consolas", 12, QFont.Weight.Bold))
        horizontalLayout_header.addWidget(self.lbl_clock)

        verticalLayout_right_container.addWidget(self.header_frame)

        # 中间内容区 (左侧视频 + 右侧控制面板)
        horizontalLayout_content = QHBoxLayout()
        horizontalLayout_content.setSpacing(16)

        # 左侧：视频区域卡片
        self.video_card = QFrame(self.right_container)
        verticalLayout_video = QVBoxLayout(self.video_card)
        verticalLayout_video.setContentsMargins(10, 10, 10, 10)
        verticalLayout_video.setSpacing(6)

        horizontalLayout_vheader = QHBoxLayout()
        self.lbl_live_badge = QLabel("未启动监控", self.video_card)
        self.lbl_live_badge.setFont(QFont("Microsoft YaHei", 11, QFont.Weight.Bold))
        horizontalLayout_vheader.addWidget(self.lbl_live_badge)
        horizontalLayout_vheader.addStretch()
        verticalLayout_video.addLayout(horizontalLayout_vheader)

        self.video_label = QLabel("请选择视频源，点击右侧【启动检测】", self.video_card)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        verticalLayout_video.addWidget(self.video_label)

        horizontalLayout_content.addWidget(self.video_card, stretch=5)

        # 右侧：控制面板与指标卡片
        self.control_card = QFrame(self.right_container)
        self.control_card.setMaximumWidth(380)
        verticalLayout_control = QVBoxLayout(self.control_card)
        verticalLayout_control.setSpacing(12)
        verticalLayout_control.setContentsMargins(16, 16, 16, 16)

        self.ctrl_title = QLabel("操作与控制台", self.control_card)
        self.ctrl_title.setFont(QFont("Microsoft YaHei", 12, QFont.Weight.Bold))
        verticalLayout_control.addWidget(self.ctrl_title)

        horizontalLayout_btn_group = QHBoxLayout()
        horizontalLayout_btn_group.setSpacing(10)

        self.btn_start = QPushButton("启动检测", self.control_card)
        horizontalLayout_btn_group.addWidget(self.btn_start)

        self.btn_stop = QPushButton("停止检测", self.control_card)
        self.btn_stop.setEnabled(False)
        horizontalLayout_btn_group.addWidget(self.btn_stop)
        verticalLayout_control.addLayout(horizontalLayout_btn_group)

        self.line_control = QFrame(self.control_card)
        self.line_control.setFrameShape(QFrame.Shape.HLine)
        self.line_control.setFixedHeight(1)
        verticalLayout_control.addWidget(self.line_control)

        self.stat_title = QLabel("实时监控指标", self.control_card)
        self.stat_title.setFont(QFont("Microsoft YaHei", 12, QFont.Weight.Bold))
        verticalLayout_control.addWidget(self.stat_title)

        self.alarm_card = QFrame(self.control_card)
        verticalLayout_alarm = QVBoxLayout(self.alarm_card)
        verticalLayout_alarm.setContentsMargins(8, 8, 8, 8)

        self.lbl_alarm_status = QLabel("系统待命", self.alarm_card)
        self.lbl_alarm_status.setFont(QFont("Microsoft YaHei", 13, QFont.Weight.Bold))
        self.lbl_alarm_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        verticalLayout_alarm.addWidget(self.lbl_alarm_status)
        verticalLayout_control.addWidget(self.alarm_card)

        horizontalLayout_subdata = QHBoxLayout()
        self.lbl_posture = QLabel("姿态: --", self.control_card)
        self.lbl_posture.setStyleSheet("font-size: 13px; font-weight: 600; border: none;")
        horizontalLayout_subdata.addWidget(self.lbl_posture)

        horizontalLayout_subdata.addStretch()

        self.lbl_fall_count = QLabel("跌倒告警: 0 次", self.control_card)
        self.lbl_fall_count.setStyleSheet("font-size: 13px; font-weight: 600; border: none;")
        horizontalLayout_subdata.addWidget(self.lbl_fall_count)
        verticalLayout_control.addLayout(horizontalLayout_subdata)

        self.log_title = QLabel("系统运行日志", self.control_card)
        self.log_title.setFont(QFont("Microsoft YaHei", 11, QFont.Weight.Bold))
        verticalLayout_control.addWidget(self.log_title)

        self.log_list = QListWidget(self.control_card)
        verticalLayout_control.addWidget(self.log_list)

        horizontalLayout_content.addWidget(self.control_card)
        verticalLayout_right_container.addLayout(horizontalLayout_content)

        horizontalLayout_main.addWidget(self.right_container)
        MainWindow.setCentralWidget(self.centralwidget)

        # 应用默认主题（ONE DARK）
        self.current_theme_name = "one_dark"
        self.current_theme = THEME_REGISTRY["one_dark"]
        self.apply_theme("one_dark")

    def apply_theme(self, theme_name: str):
        """一键切换主题：重设全局 QSS + 所有控件样式"""
        t = THEME_REGISTRY.get(theme_name)
        if not t:
            return
        self.current_theme_name = theme_name
        self.current_theme = t

        # 1. 重设全局样式表
        main_window = self.centralwidget.parent()
        if main_window:
            main_window.setStyleSheet(generate_global_stylesheet(t))

        # 2. 侧边栏
        is_glass = t["glass_opacity"] > 0
        if is_glass:
            apply_glass_effect(self.sidebar_frame, t)
        else:
            self.sidebar_frame.setStyleSheet(f"""
                QFrame {{
                    background-color: {t['bg_sidebar']};
                    border-right: 1px solid {t['sidebar_border']};
                }}
            """)

        # 3. 侧边栏标题
        self.sidebar_title.setText(t["name"])
        self.sidebar_title.setStyleSheet(f"color: {t['text_title']}; padding-left: 6px; border: none;")

        # 4. 侧边栏分隔线
        self.line_sidebar.setStyleSheet(f"background-color: {t['border']}; border: none; max-height: 1px; margin: 4px 0;")

        # 5. 顶部栏
        if is_glass:
            apply_glass_effect(self.header_frame, t)
        else:
            self.header_frame.setStyleSheet(f"""
                QFrame {{
                    background-color: {t['bg_card']};
                    border: 1px solid {t['border']};
                    border-radius: 8px;
                }}
            """)
        self.title_label.setStyleSheet(f"color: {t['text_gold']}; border: none;")
        self.lbl_clock.setStyleSheet(f"color: {t['text_clock']}; border: none;")

        # 6. 视频卡片
        if is_glass:
            apply_glass_effect(self.video_card, t)
        else:
            self.video_card.setStyleSheet(f"""
                QFrame {{
                    background-color: {t['bg_card']};
                    border: 1px solid {t['border']};
                    border-radius: 8px;
                }}
            """)
        self.lbl_live_badge.setStyleSheet(f"color: {t['text_secondary']}; border: none;")
        self.video_label.setStyleSheet(
            f"background-color: {t['bg_video']}; border-radius: 6px; "
            f"color: {t['text_video_hint']}; font-size: 14px; border: none;"
        )

        # 7. 控制面板
        if is_glass:
            apply_glass_effect(self.control_card, t)
        else:
            self.control_card.setStyleSheet(f"""
                QFrame {{
                    background-color: {t['bg_card']};
                    border: 1px solid {t['border']};
                    border-radius: 8px;
                }}
            """)
        self.ctrl_title.setStyleSheet(f"color: {t['text_title']}; border: none;")
        self.stat_title.setStyleSheet(f"color: {t['text_title']}; border: none;")
        self.log_title.setStyleSheet(f"color: {t['text_secondary']}; border: none;")

        # 8. 控制按钮（启动/停止）
        self.btn_start.setStyleSheet(f"""
            QPushButton {{
                background-color: {t['success']};
                color: {t['bg_main']};
                border: none;
                border-radius: 6px;
                font-weight: bold;
                padding: 10px;
            }}
            QPushButton:hover {{
                background-color: {t['success_hover']};
            }}
        """)
        self.btn_stop.setStyleSheet(f"""
            QPushButton {{
                background-color: {t['danger']};
                color: {t['bg_main']};
                border: none;
                border-radius: 6px;
                font-weight: bold;
                padding: 10px;
            }}
            QPushButton:hover {{
                background-color: {t['danger_hover']};
            }}
            QPushButton:disabled {{
                background-color: {t['bg_disabled']};
                color: {t['text_secondary']};
            }}
        """)

        # 9. 控制面板分隔线
        self.line_control.setStyleSheet(f"background-color: {t['border']}; border: none; max-height: 1px;")

        # 10. 告警卡片
        if is_glass:
            self.alarm_card.setStyleSheet(f"""
                QFrame {{
                    background-color: {t['bg_alarm']};
                    border: 1px solid {t['border']};
                    border-radius: 6px;
                }}
            """)
        else:
            self.alarm_card.setStyleSheet(
                f"background-color: {t['bg_alarm']}; border: 1px solid {t['border']}; border-radius: 6px;"
            )
        self.lbl_alarm_status.setStyleSheet(f"color: {t['text_title']}; border: none;")

        # 11. 姿态/告警计数文字
        self.lbl_posture.setStyleSheet(f"font-size: 13px; font-weight: 600; color: {t['text_primary']}; border: none;")
        self.lbl_fall_count.setStyleSheet(f"font-size: 13px; font-weight: 600; color: {t['danger']}; border: none;")

        # 12. 日志列表
        if is_glass:
            self.log_list.setStyleSheet(f"""
                QListWidget {{
                    background-color: {t['bg_log']};
                    border: 1px solid {t['border']};
                    border-radius: 6px;
                    color: {t['text_log']};
                    font-size: 12px;
                    padding: 4px;
                }}
            """)
        else:
            self.log_list.setStyleSheet(f"""
                QListWidget {{
                    background-color: {t['bg_log']};
                    border: 1px solid {t['border']};
                    border-radius: 6px;
                    color: {t['text_log']};
                    font-size: 12px;
                    padding: 4px;
                }}
            """)

        # 13. 主题下拉框自身样式
        # 弹出层强制使用不透明背景（bg_main 永远是纯色），避免毛玻璃主题下重影
        popup_bg = t['bg_main']
        popup_sel = t['combo_selection_bg'] if not t['combo_selection_bg'].startswith('rgba') else t['bg_btn_hover'] if not t['bg_btn_hover'].startswith('rgba') else t['border']
        self.theme_combo.setStyleSheet(f"""
            QComboBox {{
                background-color: {t['bg_input']};
                border: 1px solid {t['border_light']};
                border-radius: 6px;
                color: {t['text_primary']};
                font-size: 12px;
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
                font-size: 12px;
                padding: 4px;
            }}
        """)
