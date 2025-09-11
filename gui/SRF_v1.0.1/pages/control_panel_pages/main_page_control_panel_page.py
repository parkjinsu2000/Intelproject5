import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *
from PyQt5.QtCore import *

from pages.control_panel_pages.background_video_play_control_panel import BackgroundVideoPlayControlPanel

class MainPageControlPanelPage(QWidget):
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

        self.set_screen(screen_index)

        # 창이 실제로 그려진 뒤 버튼 위치 세팅 (한 번만 실행)
        QTimer.singleShot(0, self.set_button_location)

    def set_button_location(self):
        # 비디오 위젯 크기
        w, h = self.video_panel.video_widget.width(), self.video_panel.video_widget.height()
        bw, bh = self.single_mode_Btn.width(), self.single_mode_Btn.height()

        # 정중앙 좌표
        x = (w - bw) // 2
        y = (h - bh) // 2

        # 배치
        self.single_mode_Btn.move(x, y + bh + 200)
        self.multiple_mode_Btn.move(x, y + bh + 300)

    def set_screen(self, screen_index):
        screens = QGuiApplication.screens()
        if len(screens) > screen_index:
            geo = screens[screen_index].geometry()
            self.setGeometry(geo)
        else:
            print("지정한 모니터가 없음 → 기본 모니터 사용")



if __name__ == "__main__":
    app = QApplication(sys.argv)

    # 동영상 파일 경로 (직접 바꿔줘야 함)
    video_file = "/home/ubuntu/workspace_intel/Intelproject5/gui/jeongtae/SRF_v1.0.0/background_video_large.mp4"

    # 두 번째 모니터(1920x1080)에 띄우기
    window = MainPageControlPanelPage(video_path=video_file, screen_index=1)
    # window.showFullScreen()  # 전체화면 모드
    window.showMaximized()

    sys.exit(app.exec_())