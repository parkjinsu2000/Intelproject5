import sys
import os
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *
from PyQt5.QtCore import *
from background_video_play_control_pannel import BackgroundVideoPlayControlPannel

class AvatarSelectControlPannelPage(QWidget):
    def __init__(self, video_path=None, screen_index=0):
        super().__init__()
        
        # 메인 레이아웃 (수직)
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        
        # 동영상 기능을 가진 패널을 포함
        self.video_panel = BackgroundVideoPlayControlPannel(video_path, screen_index)
        self.main_layout.addWidget(self.video_panel)
        
        # 버튼 생성
        self.single_mode_Btn = QPushButton("변환하기", self.video_panel.video_widget)
        self.multiple_mode_Btn = QPushButton("메인으로", self.video_panel.video_widget)
        self.left_arrow_Btn = QPushButton("<", self.video_panel.video_widget)
        self.right_arrow_Btn = QPushButton(">", self.video_panel.video_widget)

        # 화살표 버튼 스타일 설정
        arrow_font = QFont()
        arrow_font.setPointSize(24)
        self.left_arrow_Btn.setFont(arrow_font)
        self.right_arrow_Btn.setFont(arrow_font)
        
        arrow_size = 60
        self.left_arrow_Btn.setFixedSize(arrow_size, arrow_size)
        self.right_arrow_Btn.setFixedSize(arrow_size, arrow_size)

        # 창이 실제로 그려진 뒤 버튼 위치 세팅 (한 번만 실행)
        QTimer.singleShot(0, self.set_button_location)
        
        # 썸네일 버튼 그리드 레이아웃
        self.grid_layout = QGridLayout()
        self.grid_layout.setHorizontalSpacing(20)
        self.grid_layout.setVerticalSpacing(20)

        # 지정된 화면에 창 배치
        self.set_screen(screen_index)
    
    def set_button_location(self):
        # 비디오 위젯 크기
        w, h = self.video_panel.video_widget.width(), self.video_panel.video_widget.height()
        
        # 메인 버튼 크기
        btn_w, btn_h = self.single_mode_Btn.width(), self.single_mode_Btn.height()
        
        # 화살표 버튼 크기
        arrow_size = self.left_arrow_Btn.width()

        # 메인 버튼 위치
        single_x = (w - btn_w) // 2
        multiple_x = (w - btn_w) // 2 + btn_w + 50
        y = (h - btn_h) // 2 + btn_h + 200
        
        self.single_mode_Btn.move(single_x, y)
        self.multiple_mode_Btn.move(multiple_x, y)

        # 화살표 버튼 위치
        arrow_y = y - arrow_size - 20
        self.left_arrow_Btn.move(single_x + (btn_w - arrow_size) // 2, arrow_y)
        self.right_arrow_Btn.move(multiple_x + (btn_w - arrow_size) // 2, arrow_y)

    def set_screen(self, screen_index):
        screens = QGuiApplication.screens()
        if len(screens) > screen_index:
            geo = screens[screen_index].geometry()
            self.setGeometry(geo)
        else:
            print("지정한 모니터가 없음 → 기본 모니터 사용")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    video_file = "/home/ubuntu/Qt/SRF_v1.0.0/resource/background_video_large.mp4"

    # 동영상 파일이 존재하는지 확인
    if not os.path.exists(video_file):
        QMessageBox.critical(None, "오류", "지정된 동영상 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        sys.exit(1)

    window = AvatarSelectControlPannel(video_path=video_file, screen_index=0)
    window.showMaximized()

    sys.exit(app.exec_())