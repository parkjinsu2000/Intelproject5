import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *
from PyQt5.QtCore import *

"""
아바타 로딩 완료 후
캐릭터가 변환된 영상을 보여주는 페이지
"""

class ConvertedAvatarVideoViewPannelPage(QWidget):
    def __init__(self, video_path=None, screen_index=1):
        super().__init__()
        # 메인 레이아웃
        self.mainLayoutV = QVBoxLayout(self)
        self.mainLayoutV.setContentsMargins(0, 0, 0, 0)

        # 그래픽스 뷰/씬
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setFrameStyle(0)
        self.mainLayoutV.addWidget(self.view)

        # 비디오 아이템
        self.video_item = QGraphicsVideoItem()
        self.video_item.setAspectRatioMode(Qt.KeepAspectRatioByExpanding)
        self.scene.addItem(self.video_item)

        # 플레이어
        self.player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.player.setVideoOutput(self.video_item)

        self.init_video(video_path)
        self.set_screen(screen_index)

        # # 창 뜬 직후 비디오/이미지 크기 세팅
        QTimer.singleShot(0, self.fit_scene)

    def fit_scene(self):
        rect = QRectF(self.view.viewport().rect())
        # 비디오를 뷰포트 전체로 맞춤
        self.scene.setSceneRect(rect)
        self.video_item.setSize(rect.size())

    def init_video(self, video_path):
        if video_path:
            self.player.setMedia(QMediaContent(QUrl.fromLocalFile(video_path)))
            self.player.setVolume(0)
            self.player.play()

    def set_screen(self, screen_index):
        screens = QGuiApplication.screens()
        if len(screens) > screen_index:
            self.setGeometry(screens[screen_index].geometry())
        else:
            print("지정한 모니터가 없음 → 기본 모니터 사용")



if __name__ == "__main__":
    app = QApplication(sys.argv)

    video_file = "/home/ubuntu/workspace_intel/Intelproject5/gui/jeongtae/SRF_v1.0.0/background_video_large.mp4"
    # image_file = "/home/ubuntu/workspace_intel/Intelproject5/gui/jeongtae/SRF_v1.0.0/title_image_4.png"

    window = ConvertedAvatarVideoViewPannelPage(video_path=video_file, screen_index=0)
    window.showMaximized()
    sys.exit(app.exec_())