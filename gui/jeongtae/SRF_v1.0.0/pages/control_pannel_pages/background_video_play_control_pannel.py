import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *
from PyQt5.QtCore import *


class BackgroundVideoPlayControlPannel(QWidget):
    def __init__(self, video_path=None, screen_index=0):
        super().__init__()

        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)

        # 비디오 출력 위젯
        self.video_widget = QVideoWidget(self)
        self.layout().addWidget(self.video_widget)

        # 플레이어
        self.player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.player.setVideoOutput(self.video_widget)

        self.init_video(video_path)
        self.set_screen(screen_index)

    def init_video(self, video_path):
        if video_path:
            self.player.setMedia(QMediaContent(QUrl.fromLocalFile(video_path)))
            self.player.setVolume(0)
            self.player.play()

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
    window = BackgroundVideoPlayControlPannel(video_path=video_file, screen_index=1)
    # window.showFullScreen()  # 전체화면 모드
    window.showMaximized()

    sys.exit(app.exec_())