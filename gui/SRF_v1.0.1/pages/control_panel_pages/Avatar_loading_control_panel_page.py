import sys
import os
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *
from PyQt5.QtCore import *
from pages.control_panel_pages.background_video_play_control_panel import BackgroundVideoPlayControlPanel

class AvatarLoadingControlPanelPage(QWidget):
    def __init__(self, video_path=None, screen_index=0):
        super().__init__()
        
        # 메인 레이아웃을 사용하지 않고, 배경 동영상 패널만 추가
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        
        self.video_panel = BackgroundVideoPlayControlPanel(video_path, screen_index)
        self.main_layout.addWidget(self.video_panel)
        
        # '변환된 영상 보기' 버튼 생성
        # 부모 위젯을 self (메인 윈도우)로 변경
        self.view_converted_btn = QPushButton("변환된 영상 보기 (활성, 비활성)", self)
        self.view_converted_btn.setEnabled(False) # 처음에는 비활성화
        self.view_converted_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(93, 153, 198, 200);
                color: white;
                border-radius: 10px;
                padding: 15px;
                font-size: 18px;
            }
            QPushButton:hover {
                background-color: rgba(74, 122, 159, 200);
            }
            QPushButton:disabled {
                background-color: rgba(160, 160, 160, 150);
            }
        """)

        # 창이 실제로 그려진 뒤 버튼 위치 세팅
        QTimer.singleShot(0, self.set_button_location)
        
        # 창 크기 변경 시 버튼 위치를 다시 계산하도록 이벤트 연결
        self.resizeEvent = self.on_resize

    def set_button_location(self):
        """Sets the location of the button to the center of the window."""
        w, h = self.width(), self.height()
        btn_w, btn_h = self.view_converted_btn.width(), self.view_converted_btn.height()

        x = (w - btn_w) // 2
        y = (h - btn_h) // 2
        
        self.view_converted_btn.move(x, y)

    def on_resize(self, event):
        """Recalculates button position on window resize."""
        self.set_button_location()
        super().resizeEvent(event)

    def set_screen(self, screen_index):
        """Sets the window to a specific screen."""
        screens = QApplication.screens()
        if len(screens) > screen_index:
            geo = screens[screen_index].geometry()
            self.setGeometry(geo)
        else:
            print("지정한 모니터가 없어 기본 모니터 사용")

    def enable_button(self, enable=True):
        """Enables or disables the 'View Converted Video' button."""
        self.view_converted_btn.setEnabled(enable)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    video_file = "/home/ubuntu/Qt/SRF_v1.0.0/resource/background_video_large.mp4"

    window = AvatarLoadingControlPanelPage(video_path=video_file, screen_index=0)
    window.showMaximized()
    
    # 버튼 활성화/비활성화 테스트
    QTimer.singleShot(3000, lambda: window.enable_button(True))
    QTimer.singleShot(6000, lambda: window.enable_button(False))

    sys.exit(app.exec_())