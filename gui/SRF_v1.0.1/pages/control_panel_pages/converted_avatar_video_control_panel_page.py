import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *
from PyQt5.QtCore import *

from pages.control_panel_pages.background_video_play_control_panel import BackgroundVideoPlayControlPanel

"""
아바타 로딩 완료 후
캐릭터가 변환된 영상을 보여주는 페이지의 컨트롤 패널
"""

<<<<<<< HEAD:gui/SRF_v1.0.1/pages/control_panel_pages/converted_avatar_video_control_panel_page.py
class ConvertedAvatarVideoControlPanelPage(QWidget):
    def __init__(self, video_path=None, screen_index=0):
=======
class ConvertedAvatarVideoControlPannelPage(QWidget):
    def __init__(self, video_path=None, screen_index=):
>>>>>>> 232d96d606b13b5b5a38cea1d7d1258e10a80353:gui/SRF_v1.0.1/pages/control_pannel_pages/converted_avatar_video_control_pannel_page.py
        super().__init__()

        # 메인 레이아웃
        self.mainLayoutV = QVBoxLayout(self)
        self.mainLayoutV.setContentsMargins(0, 0, 0, 0)

        # 비디오 패널 포함
        self.video_panel = BackgroundVideoPlayControlPanel(video_path, screen_index)
        self.mainLayoutV.addWidget(self.video_panel)

        # 버튼
        self.goToMainButton = QPushButton("메인으로", self.video_panel.video_widget)

        self.set_screen(screen_index)

        # 창이 실제로 그려진 뒤 버튼 위치 세팅 (한 번만 실행)
        QTimer.singleShot(0, self.set_button_location)

    def set_button_location(self):
        # 비디오 위젯 크기
        w, h = self.video_panel.video_widget.width(), self.video_panel.video_widget.height()
        bw, bh = self.goToMainButton.width(), self.goToMainButton.height()

        # 정중앙 좌표
        x = (w - bw) // 2
        y = (h - bh) // 2

        # 배치
        self.goToMainButton.move(x, y + bh + 200)

    def set_screen(self, screen_index):
        screens = QGuiApplication.screens()
        if len(screens) > screen_index:
            geo = screens[screen_index].geometry()
            self.move(geo.x(), geo.y())   # 보조 모니터의 시작 좌표로 이동
            self.resize(geo.width(), geo.height())  # 화면 크기 맞춤
        else:
            print("지정한 모니터가 없음 → 기본 모니터 사용")




if __name__ == "__main__":
    app = QApplication(sys.argv)

    # 동영상 파일 경로 (직접 바꿔줘야 함)
    video_file = "/home/ubuntu/workspace_intel/Intelproject5/gui/jeongtae/SRF_v1.0.0/background_video_large.mp4"

    # 두 번째 모니터(1920x1080)에 띄우기
    window = ConvertedAvatarVideoControlPanelPage(video_path=video_file, screen_index=1)
    # window.showFullScreen()  # 전체화면 모드
    window.showMaximized()

    sys.exit(app.exec_())