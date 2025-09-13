import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from config import *

from pages.control_pannel_pages import *
from pages.view_pannel_pages import *


# -------------------------------
# Control 윈도우
# -------------------------------
class ControlWindow(QMainWindow):
    # Control에서 발생하는 이벤트 시그널 정의
    singleModeRequested = pyqtSignal()
    multiModeRequested = pyqtSignal()

    def __init__(self, video_path=None):
        super().__init__()
        # self.setWindowTitle("Control Panel")

        # 예시: 메인 컨트롤 페이지를 기본으로 표시
        self.mainPage = MainPageControlPanelPage(video_path=video_path, screen_index=0)
        self.setCentralWidget(self.mainPage)

        # 버튼 이벤트 연결 → 시그널 발행
        self.mainPage.single_mode_Btn.clicked.connect(self.singleModeRequested.emit)
        self.mainPage.multiple_mode_Btn.clicked.connect(self.multiModeRequested.emit)


# -------------------------------
# View 윈도우
# -------------------------------
class ViewWindow(QMainWindow):
    def __init__(self, video_path=None, image_path=None):
        super().__init__()
        # self.setWindowTitle("View Panel")

        # 예시: 메인 뷰 페이지를 기본으로 표시
        self.mainPage = MainPageViewPanelPage(video_path=video_path,
                                              image_path=image_path,
                                              screen_index=1)
        self.setCentralWidget(self.mainPage)

    # 슬롯 메서드들
    def start_single_mode(self):
        print("[View] 싱글 모드 실행")
        self.setWindowTitle("View Panel - Single Mode")

    def start_multi_mode(self):
        print("[View] 멀티 모드 실행")
        self.setWindowTitle("View Panel - Multi Mode")


# -------------------------------
# 메인 컨트롤러
# -------------------------------
class MainController(QObject):
    def __init__(self, video_path, image_path):
        super().__init__()

        # 두 윈도우 생성
        self.control = ControlWindow(video_path=video_path)
        self.view = ViewWindow(video_path=video_path, image_path=image_path)

        # 시그널 연결
        self.control.singleModeRequested.connect(self.view.start_single_mode)
        self.control.multiModeRequested.connect(self.view.start_multi_mode)

        # ----------------------------
        # 모니터 배치 (View는 메인, Control은 보조)
        # ----------------------------
        screens = QGuiApplication.screens()

        # View 배치
        if len(screens) > MonitorIndex.VIEW:
            self.view.setGeometry(screens[MonitorIndex.VIEW].geometry())

        # Control 배치
        if len(screens) > MonitorIndex.CONTROL:
            self.control.setGeometry(screens[MonitorIndex.CONTROL].geometry())


        # 창 띄우기
        self.control.show()
        self.view.show()
        # self.control.showFullScreen()
        # self.view.showFullScreen()



# -------------------------------
# 실행 엔트리포인트
# -------------------------------
def main():
    app = QApplication(sys.argv)

    video_file = "/home/ubuntu/Qt/SRF_v1.0.1/resource/background_video_large.mp4"
    image_file = "/home/ubuntu/Qt/SRF_v1.0.1/resource/title.png"


    controller = MainController(video_path=video_file, image_path=image_file)

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()