"""测试健康大屏功能"""
import sys
from PyQt6.QtWidgets import QApplication
from deidao.dialogs import HealthDashboardDialog

def main():
    app = QApplication(sys.argv)
    dialog = HealthDashboardDialog()
    dialog.show()
    print("✅ 健康大屏已启动，请查看窗口")
    print("功能测试：")
    print("  1. 数据每 2 秒自动刷新")
    print("  2. 点击任意卡片查看趋势图")
    print("  3. 异常数据会显示红色并播放告警音")
    print("  4. 点击'导出健康报告'可保存 CSV")
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
