import sys
import os
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *
from PyQt5.QtCore import *
from background_video_play_control_pannel import BackgroundVideoPlayControlPannel

class ScoreControlPannelPage(QWidget):
    def __init__(self, video_path=None, screen_index=0):
        super().__init__()
        
        # 메인 레이아웃 (수직)
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        
        # 동영상 패널
        self.video_panel = BackgroundVideoPlayControlPannel(video_path, screen_index)
        self.main_layout.addWidget(self.video_panel)

        # 버튼들을 담을 수평 레이아웃
        self.button_layout = QHBoxLayout()
        self.button_layout.setAlignment(Qt.AlignCenter)
        self.button_layout.setSpacing(20) # 버튼 간 간격 설정

        # '아바타' 버튼
        self.avatar_btn = QPushButton("아바타", self)
        self.avatar_btn.setObjectName("avatar_btn")
        self.avatar_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(93, 153, 198, 200);
                color: white;
                border-radius: 10px;
                padding: 15px 30px;
                font-size: 18px;
            }
            QPushButton:hover {
                background-color: rgba(74, 122, 159, 200);
            }
        """)

        # '메인으로' 버튼
        self.main_btn = QPushButton("메인으로", self)
        self.main_btn.setObjectName("main_btn")
        self.main_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(93, 153, 198, 200);
                color: white;
                border-radius: 10px;
                padding: 15px 30px;
                font-size: 18px;
            }
            QPushButton:hover {
                background-color: rgba(74, 122, 159, 200);
            }
        """)

        # '다시하기' 버튼
        self.retry_btn = QPushButton("다시하기", self)
        self.retry_btn.setObjectName("retry_btn")
        self.retry_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(93, 153, 198, 200);
                color: white;
                border-radius: 10px;
                padding: 15px 30px;
                font-size: 18px;
            }
            QPushButton:hover {
                background-color: rgba(74, 122, 159, 200);
            }
        """)

        # 버튼들을 수평 레이아웃에 추가
        self.button_layout.addWidget(self.avatar_btn)
        self.button_layout.addWidget(self.main_btn)
        self.button_layout.addWidget(self.retry_btn)
        
        # 버튼 레이아웃을 메인 레이아웃에 추가
        self.main_layout.addLayout(self.button_layout)


    def set_screen(self, screen_index):
        """Sets the window to a specific screen."""
        screens = QApplication.screens()
        if len(screens) > screen_index:
            geo = screens[screen_index].geometry()
            self.setGeometry(geo)
        else:
            print("지정한 모니터가 없어 기본 모니터 사용")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    video_file = "/home/ubuntu/Qt/SRF_v1.0.0/resource/background_video_large.mp4"

    window = ScoreControlPannelPage(video_path=video_file, screen_index=0)
    window.showMaximized()
    
    sys.exit(app.exec_())
