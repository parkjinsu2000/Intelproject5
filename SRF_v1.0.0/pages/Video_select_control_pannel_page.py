import sys
import os
from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QPushButton
from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget

class VideoSelectControlPannelPage(QWidget):
    def __init__(self, video_path=None, screen_index=0): # 기본값을 0으로 변경
        super().__init__()
        self.setWindowTitle(' ')

        self.video_widget = QVideoWidget(self)

        self.buttons_container = QWidget(self)
        buttons_layout = QGridLayout(self.buttons_container)
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        buttons_layout.setSpacing(0)

        self.player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.player.setVideoOutput(self.video_widget)
        
        # Add error signal connection
        self.player.error.connect(self.handle_media_error)
        
        if video_path:
            # Converting to absolute path
            absolute_video_path = os.path.abspath(video_path)
            
            # Print the path for debugging
            print(f"Attempting to load video from: {absolute_video_path}")
            
            if not os.path.exists(absolute_video_path):
                print("Error: The file path does not exist.")
            else:
                self.player.setMedia(QMediaContent(QUrl.fromLocalFile(absolute_video_path)))
                self.player.setVolume(0)
                self.player.play()
                self.player.mediaStatusChanged.connect(self.loop_video)
        
        thumbnail_style = """
            QPushButton {
                background-color: rgba(173, 216, 230, 0.8);
                font-size: 16px;
                border: none;
            }
            QPushButton:hover {
                background-color: rgba(169, 208, 230, 0.9);
            }
        """
        
        name_style = """
            QPushButton {
                background-color: rgba(70, 130, 180, 0.9);
                color: white;
                font-size: 14px;
                border: none;
                margin-bottom: 40px;
            }
            QPushButton:hover {
                background-color: rgba(60, 103, 147, 0.95);
            }
        """

        items_per_row = 3
        num_items = 6

        for i in range(num_items):
            row = i // items_per_row
            col = i % items_per_row
            
            thumbnail_button = QPushButton(f'썸네일 {i+1}')
            thumbnail_button.setFixedSize(500, 400)
            thumbnail_button.setStyleSheet(thumbnail_style)
            
            name_button = QPushButton(f'이름 {i+1}')
            name_button.setFixedSize(500, 100)
            name_button.setStyleSheet(name_style)
            
            buttons_layout.addWidget(thumbnail_button, row * 2, col, 1, 1)
            buttons_layout.addWidget(name_button, row * 2 + 1, col, 1, 1)

            thumbnail_button.clicked.connect(lambda _, item_num=i+1: self.on_item_clicked(f'썸네일 {item_num}'))
            name_button.clicked.connect(lambda _, item_num=i+1: self.on_item_clicked(f'이름 {item_num}'))
            
        start_button = QPushButton('시작하기')
        start_button.setStyleSheet("""
            QPushButton {
                background-color: royalblue;
                color: white;
                font-size: 20px;
                padding: 15px 40px;
                border-radius: 5px;
                border: none;
                margin-top: 20px;
            }
            QPushButton:hover {
                background-color: dodgerblue;
            }
        """)
        
        buttons_layout.addWidget(start_button, (num_items // items_per_row) * 2 + 2, 1, 1, 1, Qt.AlignCenter)
        
        self.set_screen(screen_index)

    def resizeEvent(self, event):
        self.video_widget.setGeometry(self.rect())
        self.buttons_container.setGeometry(self.rect())
        super().resizeEvent(event)

    def loop_video(self, status):
        if status == QMediaPlayer.EndOfMedia:
            self.player.setPosition(0)
            self.player.play()

    def on_item_clicked(self, item_name):
        print(f'{item_name} 버튼이 클릭되었습니다.')

    def set_screen(self, screen_index):
        screens = QApplication.screens()
        # 유효한 인덱스가 아니면 기본 모니터(0번)를 사용하도록 수정
        if 0 <= screen_index < len(screens):
            target_screen = screens[screen_index]
            geo = target_screen.geometry()
            self.setGeometry(geo)
        else:
            print(f"지정한 모니터({screen_index})가 없습니다. → 기본 모니터(0)를 사용합니다.")
            if len(screens) > 0:
                target_screen = screens[0]
                geo = target_screen.geometry()
                self.setGeometry(geo)
            else:
                print("사용 가능한 모니터가 없습니다.")
    
    def handle_media_error(self, error):
        print(f"QMediaPlayer Error: {self.player.errorString()}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    video_file = "/home/ubuntu/Qt/SRF_v1.0.0/resource/background_video_large.mp4"

    # 파일 경로가 존재하는지 최종적으로 확인
    if not os.path.exists(video_file):
        print(f"Error: Video file not found at {video_file}")
        sys.exit(1)
    else:
        print(f"Success: Video file found at {video_file}")

    # screen_index=1로 설정되어 있어서 오류가 발생했습니다.
    # 일반적으로 모니터는 0번부터 시작하므로 0번 모니터를 사용하도록 변경합니다.
    window = VideoSelectControlPannelPage(video_path=video_file, screen_index=0)
    window.showMaximized()

    sys.exit(app.exec_())
