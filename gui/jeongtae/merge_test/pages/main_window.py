from PyQt5.QtWidgets import QMainWindow, QStackedWidget
from .main_page import MainPage
from .rank_page import RankPage
from .video_select_page import VideoSelectPage
from .page_enum import PageIndex
from .pose_score_app import PoseScoreApp

class MainWindow(QMainWindow):
    def __init__(self, model, use_half):
        super().__init__()
        self.model = model
        self.use_half = use_half

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        # 페이지 초기화
        self.mainPage = MainPage()
        self.rankPage = RankPage()
        self.videoPage = VideoSelectPage(self.stack, self.model, self.use_half)

        self.stack.addWidget(self.mainPage)       # PageIndex.MAIN
        self.stack.addWidget(self.rankPage)       # PageIndex.RANK
        self.stack.addWidget(self.videoPage)      # PageIndex.VIDEO_SELECT

        self.stack.setCurrentIndex(PageIndex.MAIN)

        # 시그널 연결
        self.mainPage.viewRankRequested.connect(
            lambda: self.stack.setCurrentIndex(PageIndex.RANK)
        )
        self.rankPage.backRequested.connect(
            lambda: self.stack.setCurrentIndex(PageIndex.MAIN)
        )
        self.mainPage.challengeStartRequested.connect(
            lambda: self.stack.setCurrentIndex(PageIndex.VIDEO_SELECT)
        )
        self.videoPage.startPoseAppRequested.connect(self.launch_pose_app)

        self.resize(1280, 720)

    def on_challenge_start(self):
        """챌린지 선택 버튼 → VideoSelectPage"""
        self.stack.setCurrentIndex(PageIndex.VIDEO_SELECT)

    def launch_pose_app(self, args):
        pose_app = PoseScoreApp(args, self.model, self.use_half)
        self.stack.addWidget(pose_app)
        self.stack.setCurrentWidget(pose_app)

        pose_app.goMainRequested.connect(lambda: self.stack.setCurrentIndex(PageIndex.MAIN))
        pose_app.goRankRequested.connect(lambda: self.stack.setCurrentIndex(PageIndex.RANK))