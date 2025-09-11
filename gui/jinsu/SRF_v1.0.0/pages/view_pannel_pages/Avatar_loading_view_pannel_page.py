import sys
import os
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QProgressBar
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import Qt, QUrl, QTimer, QCoreApplication
from PyQt5.QtGui import QFont, QPalette, QColor

class AvatarLoadingViewPannelPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 메인 레이아웃 (수직)
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        
        # 동영상 재생 영역
        self.video_widget = QVideoWidget()
        self.video_widget.setStyleSheet("background-color: black;")
        self.main_layout.addWidget(self.video_widget, 5)
        
        # 미디어 플레이어 설정
        self.media_player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.media_player.setVideoOutput(self.video_widget)

        # 로딩 바 영역
        self.loading_bar = QProgressBar(self)
        self.loading_bar.setAlignment(Qt.AlignCenter)
        self.loading_bar.setFont(QFont("Arial", 16))
        self.loading_bar.setFormat("로딩 중... %p%")
        self.loading_bar.setTextVisible(True)
        self.loading_bar.setFixedHeight(50)
        self.loading_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #555;
                border-radius: 5px;
                background-color: #f0f0f0;
                color: #555;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
            }
        """)
        self.main_layout.addWidget(self.loading_bar, 1)

        # 로딩 시뮬레이션 관련 변수 초기화
        self.progress_value = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_progress)

    def set_video(self, video_path):
        """Sets the video path and starts playback."""
        if not video_path or not os.path.exists(video_path):
            print(f"Warning: Video file not found at path: {video_path}")
            return
            
        self.media_player.setMedia(QMediaContent(QUrl.fromLocalFile(video_path)))
        self.media_player.play()
    
    def set_progress(self, value):
        """Sets the progress value of the loading bar."""
        self.loading_bar.setValue(value)

    def start_loading_simulation(self):
        """Starts a timer to simulate the loading progress."""
        self.progress_value = 0
        self.set_progress(self.progress_value)
        self.timer.start(50) # 50ms 마다 1씩 증가

    def _update_progress(self):
        """Updates the progress bar value and stops the timer when finished."""
        self.progress_value += 1
        if self.progress_value <= 100:
            self.set_progress(self.progress_value)
        else:
            self.timer.stop()
            # 로딩 완료 후 필요한 추가 동작 (예: 다음 화면으로 이동)을 여기에 추가할 수 있습니다.
            print("로딩이 완료되었습니다.")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    window = QWidget()
    window.setWindowTitle("로딩 & 비디오 패널 테스트")
    window.setGeometry(100, 100, 800, 600)
    
    loading_panel = AvatarLoadingViewPannelPage()
    main_layout = QVBoxLayout(window)
    main_layout.addWidget(loading_panel)
    
    window.show()

    # 테스트를 위한 동영상 파일 경로 설정
    test_video_file = "/home/ubuntu/Qt/SRF_v1.0.0/resource/background_video_large.mp4"
    
    loading_panel.set_video(test_video_file)
    loading_panel.start_loading_simulation()

    sys.exit(app.exec_())
