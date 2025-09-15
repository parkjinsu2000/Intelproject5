import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *
from PyQt5.QtCore import *
<<<<<<< HEAD


class BackgroundVideoPlayControlPanel(QWidget):
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
=======
from config import *

class BackgroundVideoPlayControlPanel(QWidget):
    def __init__(self, video_path=None, screen_index=0, loop=True, mute=True):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Scene & View
        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        layout.addWidget(self.view)

        # Video Item
        self.video_item = QGraphicsVideoItem()
        self.video_item.setZValue(-1)
        self.scene.addItem(self.video_item)

        # Player
        self.player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.player.setVideoOutput(self.video_item)

        # Overlay 관리
        self.overlays = []

    def add_overlay_widget(self, widget, x=0, y=0, z=1):
        proxy = self.scene.addWidget(widget)
        proxy.setZValue(z)
        proxy.setPos(x, y)
        self.overlays.append(proxy)
        return proxy
    
    def set_screen(self, screen_index):
        """지정한 모니터 전체 영역에 맞춰 패널 위치/크기 조정"""
>>>>>>> 5da7c3e167d7d8b44c0f34a65c8de19bb98b5538
        screens = QGuiApplication.screens()
        if len(screens) > screen_index:
            geo = screens[screen_index].geometry()
            self.setGeometry(geo)
        else:
<<<<<<< HEAD
            print("지정한 모니터가 없음 → 기본 모니터 사용")
=======
            print("⚠️ 지정한 모니터 없음 → 기본 모니터 사용")
>>>>>>> 5da7c3e167d7d8b44c0f34a65c8de19bb98b5538



if __name__ == "__main__":
    app = QApplication(sys.argv)

<<<<<<< HEAD
    # 동영상 파일 경로 (직접 바꿔줘야 함)
    video_file = "/home/ubuntu/Qt/SRF_v1.0.0/resource/background_video_small.mp4"

    # 두 번째 모니터(1920x1080)에 띄우기
    window = BackgroundVideoPlayControlPanel(video_path=video_file, screen_index=1)
    # window.showFullScreen()  # 전체화면 모드
    window.showMaximized()

    sys.exit(app.exec_())
=======
    # 영상 파일 경로
    video_file = "/home/ubuntu/Qt/SRF_v1.0.1/resource/background_video_pade.mp4"

    # 영상 재생 패널 생성 및 전체화면 출력
    window = BackgroundVideoPlayControlPanel(video_path=video_file, screen_index=1)
    window.showFullScreen()

    sys.exit(app.exec_())











































# class BackgroundVideoPlayControlPanel(QWidget):
#     def __init__(self, video_path=None, screen_index=0, loop=True, mute=True):
#         super().__init__()

#         # 전체 배경 검은색
#         self.setStyleSheet("background-color: black;")

#         # 레이아웃
#         layout = QVBoxLayout(self)
#         layout.setContentsMargins(0, 0, 0, 0)

#         # QGraphicsScene / QGraphicsView
#         self.scene = QGraphicsScene(self)
#         self.view = QGraphicsView(self.scene)
#         self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
#         self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
#         self.view.setFrameStyle(0)
#         layout.addWidget(self.view)

#         # 비디오 아이템
#         self.video_item = QGraphicsVideoItem()
#         self.video_item.setAspectRatioMode(Qt.KeepAspectRatioByExpanding)
#         self.scene.addItem(self.video_item)

#         # 플레이어 + 플레이리스트
#         self.playlist = QMediaPlaylist()
#         if video_path:
#             self.playlist.addMedia(QMediaContent(QUrl.fromLocalFile(video_path)))
#         self.playlist.setPlaybackMode(QMediaPlaylist.CurrentItemInLoop if loop else QMediaPlaylist.CurrentItemOnce)

#         self.player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
#         self.player.setPlaylist(self.playlist)
#         self.player.setVideoOutput(self.video_item)
#         self.player.setVolume(0 if mute else 100)
#         # self.player.play()

#         # ✅ 호환성을 위해 video_widget 속성 제공 (버튼/라벨 parent로 사용 가능)
#         self.video_widget = self.view.viewport()

#         # 화면 크기에 맞춤
#         self.fit_in_view()
#         self.view.resizeEvent = lambda event: self.fit_in_view()

#         # 화면 위치/크기
#         self.set_screen(screen_index)

#     def showEvent(self, event):
#         super().showEvent(event)
#         QTimer.singleShot(0, self.play)  # 뷰가 그려진 직후 실행

#     def fit_in_view(self):
#         """비디오 아이템을 뷰 크기에 맞춤"""
#         rect = self.view.viewport().rect()
#         self.scene.setSceneRect(QRectF(rect))          # QRect → QRectF
#         self.video_item.setSize(QSizeF(rect.size()))   # QSize → QSizeF

#     def play(self):
#         if self.player:
#             self.player.play()

#     def pause(self):
#         if self.player:
#             self.player.pause()

#     def stop(self):
#         if self.player:
#             self.player.stop()

#     def add_overlay_pixmap(self, pixmap, x=0, y=0, z=1):
#         item = self.scene.addPixmap(pixmap)
#         item.setZValue(z)
#         item.setPos(x, y)
#         return item

#     def add_overlay_widget(self, widget, x=0, y=0, z=1):
#         proxy = self.scene.addWidget(widget)
#         proxy.setZValue(z)
#         proxy.setPos(x, y)
#         return proxy

#     def set_screen(self, screen_index):
#         screens = QGuiApplication.screens()
#         if len(screens) > screen_index:
#             geo = screens[screen_index].geometry()
#             self.setGeometry(geo)
#         else:
#             print("⚠️ 지정한 모니터 없음 → 기본 모니터 사용")
>>>>>>> 5da7c3e167d7d8b44c0f34a65c8de19bb98b5538
