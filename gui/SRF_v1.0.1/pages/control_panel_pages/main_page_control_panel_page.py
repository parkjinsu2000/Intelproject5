import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *
from PyQt5.QtCore import *

<<<<<<< HEAD
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
=======
# from pages.control_panel_pages.background_video_play_control_panel import BackgroundVideoPlayControlPanel

import sys
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QApplication, QGraphicsScene, QGraphicsView
from PyQt5.QtGui import QGuiApplication
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QGraphicsVideoItem
from PyQt5.QtCore import QUrl, QTimer, pyqtSignal, QRectF


class MainPageControlPanelPage(QWidget):
    singleModeRequest = pyqtSignal()
    multipleModeRequest = pyqtSignal()

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

        if video_path:
            self.player.setMedia(QMediaContent(QUrl.fromLocalFile(video_path)))
            self.player.setVolume(0)
            self.player.play()

        # 버튼 (영상 위에 올림: parent를 view.viewport()로)
        self.single_mode_Btn = QPushButton("1인 모드", self.view.viewport())
        self.multiple_mode_Btn = QPushButton("2인 모드", self.view.viewport())

        # 버튼 크기 고정
        self.single_mode_Btn.setFixedSize(200, 60)
        self.multiple_mode_Btn.setFixedSize(200, 60)

        self.single_mode_Btn.clicked.connect(self.on_clicked_single_mode_btn)
        self.multiple_mode_Btn.clicked.connect(self.on_clicked_multiple_mode_btn)

        # 모니터 위치/크기
        self.set_screen(screen_index)

        # 창 뜬 직후 비디오/버튼 위치 세팅
        QTimer.singleShot(0, self.fit_scene)
        QTimer.singleShot(0, self.set_button_location)

    def fit_scene(self):
        rect = QRectF(self.view.viewport().rect())
        self.scene.setSceneRect(rect)
        self.video_item.setSize(rect.size())

    def on_clicked_single_mode_btn(self):
        self.singleModeRequest.emit()

    def on_clicked_multiple_mode_btn(self):
        self.multipleModeRequest.emit()

    def set_button_location(self):
        """비디오 뷰포트 크기를 기준으로 버튼 위치 계산"""
        w = self.view.viewport().width()
        h = self.view.viewport().height()
        bw = self.single_mode_Btn.width()
        bh = self.single_mode_Btn.height()

        # 중앙 좌표
>>>>>>> 5da7c3e167d7d8b44c0f34a65c8de19bb98b5538
        x = (w - bw) // 2
        y = (h - bh) // 2

        # 배치
        self.single_mode_Btn.move(x, y + bh + 200)
        self.multiple_mode_Btn.move(x, y + bh + 300)

    def set_screen(self, screen_index):
        screens = QGuiApplication.screens()
        if len(screens) > screen_index:
<<<<<<< HEAD
            geo = screens[screen_index].geometry()
            self.setGeometry(geo)
        else:
            print("지정한 모니터가 없음 → 기본 모니터 사용")
=======
            self.setGeometry(screens[screen_index].geometry())
        else:
            print("⚠️ 지정한 모니터 없음 → 기본 모니터 사용")
>>>>>>> 5da7c3e167d7d8b44c0f34a65c8de19bb98b5538



if __name__ == "__main__":
    app = QApplication(sys.argv)

    # 동영상 파일 경로 (직접 바꿔줘야 함)
    video_file = "/home/ubuntu/workspace_intel/Intelproject5/gui/jeongtae/SRF_v1.0.0/background_video_large.mp4"

    # 두 번째 모니터(1920x1080)에 띄우기
    window = MainPageControlPanelPage(video_path=video_file, screen_index=1)
    # window.showFullScreen()  # 전체화면 모드
    window.showMaximized()

    sys.exit(app.exec_())