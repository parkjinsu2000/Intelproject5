import sys
import os
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
#from background_video_play_control_panel import BackgroundVideoPlayControlPanel
from pages.control_panel_pages.background_video_play_control_panel import BackgroundVideoPlayControlPanel

class VideoSelectControlPanelPage(QWidget):
    def __init__(self, video_path=None, screen_index=1):
        super().__init__()

        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)

        # 비디오 패널 포함
        self.video_panel = BackgroundVideoPlayControlPanel(video_path, screen_index)
        self.layout().addWidget(self.video_panel)

         # 버튼
        self.single_mode_Btn = QPushButton("1인 모드", self.video_panel.video_widget)
        self.multiple_mode_Btn = QPushButton("2인 모드", self.video_panel.video_widget)

      

    def on_item_clicked(self, path):
        QMessageBox.information(self, "Video Selected", f"Selected video path:\n{path}")

    def set_screen(self, screen_index):
        screens = QApplication.screens()
        if len(screens) > screen_index:
            geo = screens[screen_index].geometry()
            self.setGeometry(geo)
        else:
            print("No monitor found at the specified index. Using default monitor.")

if __name__ == '__main__':
    try:
        app = QApplication(sys.argv)
        video_file = "/home/ubuntu/Qt/SRF_v1.0.1/resource/background_video_pade.mp4"
        window = VideoSelectControlPanelPage(video_path=video_file, screen_index=1)
        window.set_screen(1)
        window.showFullScreen()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"[CRASH] Exception occurred: {e}")
