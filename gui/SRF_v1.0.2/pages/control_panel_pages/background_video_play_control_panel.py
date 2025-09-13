import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QStackedLayout, QSizePolicy
)
from PyQt5.QtGui import QGuiApplication
from PyQt5.QtMultimedia import QMediaPlayer, QMediaPlaylist, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import Qt, QUrl

class BackgroundVideoPlayControlPanel(QWidget):
    def __init__(self, video_path=None, screen_index=0):
        super().__init__()

        self.setStyleSheet("background-color: black;")

        # QStackedLayout을 사용해 위젯을 겹칩니다.
        stacked_layout = QStackedLayout()
        # StackAll 모드는 모든 위젯을 동시에 표시하도록 설정합니다.
        stacked_layout.setStackingMode(QStackedLayout.StackAll)

        # 1. 비디오 위젯 (아래에 깔릴 배경)
        self.video_widget = QVideoWidget()
        self.video_widget.setStyleSheet("background-color: black;")
        # 비디오 위젯이 전체 공간을 차지하도록 설정합니다.
        self.video_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        stacked_layout.addWidget(self.video_widget)

        # 2. 오버레이 위젯 (위에 겹쳐질 버튼)
        overlay_widget = QWidget()
        # 마우스 이벤트를 받기 위해 투명 속성을 끕니다.
        overlay_widget.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        # 배경색을 투명하게 설정하여 아래 비디오가 보이게 합니다.
        overlay_widget.setStyleSheet("background: transparent;")
        
        overlay_layout = QVBoxLayout(overlay_widget)
        overlay_layout.setAlignment(Qt.AlignCenter)
        overlay_layout.setContentsMargins(0, 0, 0, 0)
        
        btn = QPushButton("🔥 STREET FIGHTER R")
        btn.setStyleSheet("""
            QPushButton {
                color: white;
                background-color: rgba(0, 0, 0, 0.6);
                font-size: 24px;
                padding: 12px 24px;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: rgba(255, 0, 0, 0.6);
            }
        """)
        btn.clicked.connect(self.on_button_click)
        overlay_layout.addWidget(btn)
        stacked_layout.addWidget(overlay_widget)

        # 메인 레이아웃에 QStackedLayout을 추가합니다.
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addLayout(stacked_layout)

        # 미디어 플레이어 설정
        self.player = QMediaPlayer(self)
        self.player.setVideoOutput(self.video_widget)

        self.init_video(video_path)
        self.set_screen(screen_index)
        
    def init_video(self, video_path):
        if video_path:
            self.playlist = QMediaPlaylist()
            self.playlist.addMedia(QMediaContent(QUrl.fromLocalFile(video_path)))
            self.playlist.setPlaybackMode(QMediaPlaylist.CurrentItemInLoop)

            self.player.setPlaylist(self.playlist)
            self.player.setVolume(0)
            self.player.play()
    
    def on_button_click(self):
        print("버튼이 클릭되었습니다!")

    def set_screen(self, screen_index):
        screens = QGuiApplication.screens()
        if len(screens) > screen_index:
            geo = screens[screen_index].geometry()
            self.setGeometry(geo)
        else:
            print("지정한 모니터가 없음 → 기본 모니터 사용")

if __name__ == "__main__":
    app = QApplication(sys.argv)

    video_file = "/home/ubuntu/Qt/SRF_v1.0.1/resource/background_video_pade.mp4"

    window = BackgroundVideoPlayControlPanel(video_path=video_file, screen_index=1)
    window.showFullScreen()

    sys.exit(app.exec_())