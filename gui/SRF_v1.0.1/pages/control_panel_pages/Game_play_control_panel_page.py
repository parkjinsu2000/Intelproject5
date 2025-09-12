import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *
from PyQt5.QtCore import *

from pages.control_panel_pages.background_video_play_control_panel import BackgroundVideoPlayControlPanel
# from background_video_play_control_panel import *

class GamePlayControlPanelPage(QWidget):
    def __init__(self, video_path=None, screen_index=1):
        super().__init__()

        # 메인 레이아웃
        self.mainLayoutV = QVBoxLayout(self)
        self.mainLayoutV.setContentsMargins(0, 0, 0, 0)

        # 비디오 패널 포함
        self.video_panel = BackgroundVideoPlayControlPanel(video_path, screen_index)
        self.mainLayoutV.addWidget(self.video_panel)

        # 모니터 위치/크기
        self.set_screen(screen_index)

    def showEvent(self, event):
        super().showEvent(event)

    def set_screen(self, screen_index):
        self.video_panel.set_screen(screen_index)  # ✅ video_panel의 set_screen 활용



if __name__ == "__main__":
    app = QApplication(sys.argv)

    # 동영상 파일 경로 (직접 바꿔줘야 함)
    video_file = "/home/ubuntu/workspace_intel/Intelproject5/gui/SRF_v1.0.1/resource/background_video_large.mp4"

    # 두 번째 모니터(1920x1080)에 띄우기
    window = GamePlayControlPanelPage(video_path=video_file, screen_index=1)
    # window.showFullScreen()  # 전체화면 모드
    window.showMaximized()

    sys.exit(app.exec_())