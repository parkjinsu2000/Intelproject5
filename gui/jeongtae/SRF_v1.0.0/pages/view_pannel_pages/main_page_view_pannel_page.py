import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *
from PyQt5.QtCore import *


import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel
from PyQt5.QtGui import QPixmap
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import QUrl, Qt


import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QStackedLayout
from PyQt5.QtGui import QPixmap
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import QUrl, Qt


class MainPageViewPanelPage(QWidget):
    def __init__(self, video_path=None, image_path=None, screen_index=0):
        super().__init__()

        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)

        # 비디오 출력 위젯
        self.video_widget = QVideoWidget(self)
        self.layout().addWidget(self.video_widget)

        # 플레이어
        self.player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.player.setVideoOutput(self.video_widget)

        self.top_logo_label = QLabel(self)

        # 이미지 로드
        if image_path:
            pixmap = QPixmap(image_path)
            self.top_logo_label.setPixmap(pixmap)  # 원본 크기 그대로
            self.top_logo_label.adjustSize()       # 라벨 크기를 이미지 크기에 맞춤
            self.top_logo_label.move(
                (self.width() - self.top_logo_label.width()) // 2,  # 화면 중앙 정렬 (가로)
                10  # 상단에서 10px 아래 고정
            )
            self.top_logo_label.raise_()           # 비디오 위에 보이도록
            self.top_logo_label.show()
        else:
            self.top_logo_label.hide()

        self.init_video(video_path)
        self.set_screen(screen_index)

    def init_video(self, video_path):
        if video_path:
            self.player.setMedia(QMediaContent(QUrl.fromLocalFile(video_path)))
            self.player.setVolume(0)
            self.player.play()

    def set_screen(self, screen_index):
        screens = QApplication.screens()
        if len(screens) > screen_index:
            geo = screens[screen_index].geometry()
            self.setGeometry(geo)
        else:
            print("지정한 모니터가 없음 → 기본 모니터 사용")


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # 경로 수정해서 테스트하세요
    video_file = "/home/ubuntu/workspace_intel/Intelproject5/gui/jeongtae/SRF_v1.0.0/background_video_large.mp4"
    image_file = "/home/ubuntu/workspace_intel/Intelproject5/gui/jeongtae/SRF_v1.0.0/title_image_1.png"

    window = MainPageViewPanelPage(video_path=video_file, image_path=image_file, screen_index=0)
    window.showMaximized()  # 전체화면에 맞게 표시
    # window.showFullScreen()

    sys.exit(app.exec_())