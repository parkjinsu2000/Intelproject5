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
    goToRequested = pyqtSignal()
    startRequested = pyqtSignal()

    def __init__(self, video_path=None):
        super().__init__()

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        # 페이지 추가
        self.mainPage = MainPageControlPanelPage(video_path=video_path, screen_index=0)
        self.videoSelectPage = VideoSelectControlPanelPage(video_path=video_path, screen_index=0)
        self.gamePlayControlPanelPage = GamePlayControlPanelPage(video_path=video_path, screen_index=0)
        self.scorePage = ScoreControlPanelPage(video_path=video_path, screen_index=0)

        self.stack.addWidget(self.mainPage)                         # index 0
        self.stack.addWidget(self.videoSelectPage)                  # index 1
        self.stack.addWidget(self.gamePlayControlPanelPage)         # index 2
        self.stack.addWidget(self.scorePage)                        # index 3

        # 기본 페이지
        self.stack.setCurrentIndex(ControlPageIndex.MAIN)
        # self.stack.setCurrentIndex(ControlPageIndex.GAME_PLAY)

        self.signal_connect()

    def signal_connect(self):
        # 버튼 이벤트 연결 → 시그널 발행 + 페이지 전환
        self.mainPage.singleModeRequest.connect(self.handle_single_mode)
        self.mainPage.multipleModeRequest.connect(self.handle_multi_mode)
        self.videoSelectPage.backRequested.connect(self.go_to_main)
        self.videoSelectPage.startRequested.connect(self.game_start)

    def handle_single_mode(self):
        self.singleModeRequested.emit()
        self.stack.setCurrentIndex(ControlPageIndex.VIDEO_SELECT)

    def handle_multi_mode(self):
        self.multiModeRequested.emit()
        self.stack.setCurrentIndex(ControlPageIndex.SCORE)

    def go_to_main(self):
        self.goToRequested.emit()
        self.stack.setCurrentIndex(ControlPageIndex.MAIN)

    def game_start(self):
        self.startRequested.emit()
        self.stack.setCurrentIndex(ControlPageIndex.GAME_PLAY)


# -------------------------------
# View 윈도우
# -------------------------------
class ViewWindow(QMainWindow):
    def __init__(self, video_path=None, image_path=None):
        super().__init__()

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        # 페이지 추가
        self.mainPage = MainPageViewPanelPage(video_path=video_path, image_path=image_path, screen_index=1)
        self.videoSelectViewPanelPage = VideoSelectViewPanelPage()
        self.gamePlayViewPanelPage = GamePlayViewPanelPage()
        self.scorePage = ScoreViewPanelPage(image_path=image_path)

        self.stack.addWidget(self.mainPage)                 # index 0
        self.stack.addWidget(self.videoSelectViewPanelPage) # index 1
        self.stack.addWidget(self.gamePlayViewPanelPage)    # index 2
        self.stack.addWidget(self.scorePage)                # index 3

        self.stack.setCurrentIndex(ViewPageIndex.MAIN)

    # 슬롯 메서드들 (Control 시그널에 연결됨)
    def start_single_mode(self):
        print("[View] 싱글 모드 실행 → 게임 플레이 페이지로 전환")
        self.stack.setCurrentIndex(ViewPageIndex.VIDEO_SELECT)

    def start_multi_mode(self):
        print("[View] 멀티 모드 실행 → 점수 페이지로 전환")
        # self.stack.setCurrentIndex(self.scorePage)

    def go_to_main(self):
        self.stack.setCurrentIndex(ViewPageIndex.MAIN)

    def game_start(self):
        self.stack.setCurrentIndex(ViewPageIndex.GAME_PLAY)


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
        self.control.goToRequested.connect(self.view.go_to_main)
        self.control.startRequested.connect(self.view.game_start)

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