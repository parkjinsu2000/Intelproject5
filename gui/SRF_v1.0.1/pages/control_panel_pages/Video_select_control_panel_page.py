import sys
import os
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *
from PyQt5.QtCore import *
from pages.control_panel_pages.background_video_play_control_panel import BackgroundVideoPlayControlPanel

class VideoSelectControlPanelPage(QWidget):
    def __init__(self, video_path=None, screen_index=0):
        super().__init__()
      
        # 메인 레이아웃 (수직)
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        
        # 동영상 기능을 가진 패널을 포함시킵니다.
        self.video_panel = BackgroundVideoPlayControlPanel(video_path, screen_index)
        self.main_layout.addWidget(self.video_panel)
        
        # 단일 이미지 경로
        thumbnail_image_path = "/home/ubuntu/Qt/SRF_v1.0.0/resource/button1.png"
        name_button_image_path = "/home/ubuntu/Qt/SRF_v1.0.0/resource/button.png"
        
        # 버튼들을 비디오 위젯 위에 직접 배치
        self.buttons = []
        rows = 2
        cols = 3
        
        for i in range(rows * cols):
            # 썸네일 버튼 스타일 (CSS로 이미지 적용)
            thumbnail_style = f"""
                QPushButton {{
                    background-image: url({thumbnail_image_path});
                    background-repeat: no-repeat;
                    background-position: center;
                    background-size: 100% 100%; /* 버튼 크기에 맞게 이미지 늘리기 */
                    border: none;
                    border-radius: 5px;
                }}
                QPushButton:hover {{
                    background-color: rgba(169, 208, 230, 0.9);
                }}
            """
            
            # 썸네일 버튼
            thumb_w, thumb_h = 500, 400
            thumbnail_btn = QPushButton('', self.video_panel.video_widget)
            thumbnail_btn.setFixedSize(thumb_w, thumb_h)
            thumbnail_btn.setStyleSheet(thumbnail_style)
            self.buttons.append(thumbnail_btn)
            
            # 이름 버튼 스타일 (CSS로 이미지 적용)
            name_style = f"""
                QPushButton {{
                    background-image: url({name_button_image_path});
                    background-repeat: no-repeat;
                    background-position: center;
                    background-size: 100% 100%; /* 버튼 크기에 맞게 이미지 늘리기 */
                    border: none;
                    border-radius: 5px;
                }}
                QPushButton:hover {{
                    background-color: rgba(70, 130, 180, 0.5);
                }}
            """

            # 이름 버튼
            name_w, name_h = 500, 100
            name_btn = QPushButton('', self.video_panel.video_widget)
            name_btn.setFixedSize(name_w, name_h)
            name_btn.setStyleSheet(name_style)
            self.buttons.append(name_btn)
            
            # 클릭 이벤트
            thumbnail_btn.clicked.connect(lambda _, item=i+1: self.on_item_clicked(f'썸네일 {item}'))
            name_btn.clicked.connect(lambda _, item=i+1: self.on_item_clicked(f'이름 {item}'))
        
        # '시작하기' 버튼
        self.start_button = QPushButton('시작하기', self.video_panel.video_widget)
        # 버튼의 크기를 고정된 값으로 설정
        self.start_button.setFixedSize(200, 60)
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: royalblue;
                color: white;
                font-size: 20px;
                border-radius: 5px;
                border: none;
            }
            QPushButton:hover {
                background-color: dodgerblue;
            }
        """)
        self.buttons.append(self.start_button)
        
        # 창이 실제로 그려진 뒤 버튼 위치 세팅
        QTimer.singleShot(0, self.set_buttons_location)

    def set_buttons_location(self):
        w, h = self.video_panel.video_widget.width(), self.video_panel.video_widget.height()
        
        rows = 2
        cols = 3
        thumb_width, thumb_height = 500, 400
        name_height = 100
        padding = 50
        
        total_grid_width = cols * thumb_width + (cols - 1) * padding
        total_grid_height = rows * (thumb_height + name_height) + (rows - 1) * padding + 100
        
        start_x = int((w - total_grid_width) / 2)
        start_y = int((h - total_grid_height) / 2)

        for i in range(len(self.buttons) // 2):
            row = i // cols
            col = i % cols
            
            x = start_x + col * (thumb_width + padding)
            y_thumb = start_y + row * (thumb_height + name_height + padding)
            y_name = y_thumb + thumb_height
            
            self.buttons[2*i].move(x, y_thumb)
            self.buttons[2*i+1].move(x, y_name)
        
        # '시작하기' 버튼 위치
        start_btn_width = self.start_button.width()
        self.start_button.move(int((w - start_btn_width) / 2), int(start_y + total_grid_height - 100))

    def resizeEvent(self, event):
        self.set_buttons_location()
        super().resizeEvent(event)

    def on_item_clicked(self, item_name):
        print(f'{item_name} 버튼이 클릭되었습니다.')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    video_file = "/home/ubuntu/Qt/SRF_v1.0.0/resource/background_video_large.mp4"

    window = VideoSelectControlPanelPage(video_path=video_file, screen_index=0)
    window.showMaximized()

    sys.exit(app.exec_())
