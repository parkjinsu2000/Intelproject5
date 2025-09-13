import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *
from PyQt5.QtCore import *

"""
메인 페이지
상단에 '스트릿 릴스 파이터' 문구가 뜨고
배경 영상에는 캐릭터들이 춤을 추고 있음
"""

class MainPageViewPanelPage(QWidget):
    def __init__(self, video_path=None, image_path=None, screen_index=0):
        super().__init__()

        self.image_path = image_path

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

        # 이미지 아이템
        self.image_item = None
        if image_path:
            pixmap = QPixmap(image_path)
            self.image_item = self.scene.addPixmap(pixmap)
            self.image_item.setZValue(1)  # 영상 위에 오도록
            self.image_item.setTransformationMode(Qt.SmoothTransformation)

        # 플레이어
        self.player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.player.setVideoOutput(self.video_item)

        self.init_video(video_path)
        self.set_screen(screen_index)

        # 창 뜬 직후 비디오/이미지 크기 세팅
        QTimer.singleShot(0, self.fit_scene)
        # QTimer.singleShot(0, self.set_title_image)

    def fit_scene(self):
        rect = QRectF(self.view.viewport().rect())
        # 비디오를 뷰포트 전체로 맞춤
        self.scene.setSceneRect(rect)
        self.video_item.setSize(rect.size())

        # 이미지 중앙 상단 배치 (원본 크기 그대로)
        if self.image_item:
            pixmap = QPixmap(self.image_path)  # 원본 사이즈 유지
            self.image_item.setPixmap(pixmap)

            x = (rect.width() - pixmap.width()) / 2  # 가로 중앙
            y = 0                                    # 세로 상단
            self.image_item.setPos(x, y)

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
    image_file = "/home/ubuntu/workspace_intel/Intelproject5/gui/jeongtae/SRF_v1.0.0/title_image_4.png"

    window = MainPageViewPanelPage(video_path=video_file, image_path=image_file, screen_index=0)
    window.showMaximized()
    sys.exit(app.exec_())