import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from config import *

from pages.control_panel_pages import (
    MainPageControlPanelPage,
    VideoSelectControlPanelPage,
    BackgroundVideoPlayControlPanel,
    ScoreControlPanelPage,
    AvatarSelectControlPanelPage,
    AvatarLoadingControlPanelPage,
    ConvertedAvatarVideoControlPanelPage,
)
from pages.view_pannel_pages import (
    MainPageViewPanelPage,
    VideoSelectViewPanelPage,
    GamePlayViewPanelPage,
    ScoreViewPanelPage,
    AvatarSelectViewPanelPage,
    AvatarLoadingViewPanelPage,
    ConvertedAvatarVideoViewPanelPage,
)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        # 페이지 초기화
        self.mainPageControlPanelPage = MainPageControlPanelPage()
        self.mainPageViewPanelPage = MainPageViewPanelPage()

        self.videoSelectControlPanelPage = VideoSelectControlPanelPage()
        self.videoSelectViewPanelPage = VideoSelectViewPanelPage()

        self.backgroundVideoPlayControlPanelPage = BackgroundVideoPlayControlPanel()
        self.gamePlayViewPanelPage = GamePlayViewPanelPage()

        self.scoreControlPanelPage = ScoreControlPanelPage()
        self.scoreViewPanelPage = ScoreViewPanelPage()

        self.avatarSelectControlPanelPage = AvatarSelectControlPanelPage()
        self.avatarSelectViewPanelPage = AvatarSelectViewPanelPage()

        self.avatarLoadingControlPanelPage = AvatarLoadingControlPanelPage()
        self.avatarLoadingViewPanelPage = AvatarLoadingViewPanelPage()

        self.convertedAvatarVideoControlPanelPage = ConvertedAvatarVideoControlPanelPage()
        self.convertedAvatarVideoViewPanelPage = ConvertedAvatarVideoViewPanelPage()

        self.stack.addWidget(self.mainPageControlPanelPage)
        self.stack.addWidget(self.mainPageViewPanelPage)

        self.stack.addWidget(self.videoSelectControlPanelPage)
        self.stack.addWidget(self.videoSelectViewPanelPage)

        self.stack.addWidget(self.backgroundVideoPlayControlPanelPage)
        self.stack.addWidget(self.gamePlayViewPanelPage)

        self.stack.addWidget(self.scoreControlPanelPage)
        self.stack.addWidget(self.scoreViewPanelPage)

        self.stack.addWidget(self.avatarSelectControlPanelPage)
        self.stack.addWidget(self.avatarSelectViewPanelPage)

        self.stack.addWidget(self.avatarLoadingControlPanelPage)
        self.stack.addWidget(self.avatarLoadingViewPanelPage)

        self.stack.addWidget(self.convertedAvatarVideoControlPanelPage)
        self.stack.addWidget(self.convertedAvatarVideoViewPanelPage)

        self.stack.setCurrentIndex()

        self.signal_connect()

    def signal_connect():
        pass

def main():
    # Qt 앱 실행
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()

if __name__ == "__main__":
    main()
