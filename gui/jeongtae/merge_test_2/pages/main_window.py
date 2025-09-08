from PyQt5.QtWidgets import QMainWindow, QStackedWidget
from .main_page import MainPage
from .rank_page import RankPage
from .video_select_page import VideoSelectPage
from .page_enum import PageIndex

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pose Challenge GUI (Demo)")
        self.resize(1280, 720)

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        # 페이지 초기화
        self.mainPage = MainPage()
        self.rankPage = RankPage()
        self.videoPage = VideoSelectPage(self.stack)

        self.stack.addWidget(self.mainPage)   # PageIndex.MAIN
        self.stack.addWidget(self.rankPage)   # PageIndex.RANK
        self.stack.addWidget(self.videoPage)  # PageIndex.VIDEO_SELECT

        self.stack.setCurrentIndex(PageIndex.MAIN)

        # 시그널 연결
        self.mainPage.viewRankRequested.connect(
            lambda: self.stack.setCurrentIndex(PageIndex.RANK)
        )
        self.mainPage.challengeStartRequested.connect(
            lambda: self.stack.setCurrentIndex(PageIndex.VIDEO_SELECT)
        )
        self.rankPage.backRequested.connect(
            lambda: self.stack.setCurrentIndex(PageIndex.MAIN)
        )
        # self.videoPage.backRequested.connect(
        #     lambda: self.stack.setCurrentIndex(PageIndex.MAIN)
        # )
