from PyQt5.QtWidgets import QMainWindow, QStackedWidget, QMessageBox
from .main_page import MainPage
from .rank_page import RankPage
from .video_select_page import VideoSelectPage
from .page_enum import PageIndex, ModeNumber
from .Single_Player_app import SinglePlayerApp
from .Multi_Player_app import MultiPlayerApp


class MainWindow(QMainWindow):
    def __init__(self, model, use_half):
        super().__init__()
        self.model = model
        self.use_half = use_half
        self.mode = ModeNumber.SINGLE

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
        self.mainPage.challengeStartRequested.connect(self.on_challenge_start)
        self.videoPage.startPoseAppRequested.connect(self.launch_pose_app)

        self.resize(1280, 720)

    def on_challenge_start(self, mode: ModeNumber):
        """챌린지 선택 버튼 → VideoSelectPage"""
        self.mode = mode
        self.mainPage.mode = self.mode
        print(f"[DEBUG] 선택한 모드: {self.mode}")
        self.stack.setCurrentIndex(PageIndex.VIDEO_SELECT)

    def launch_pose_app(self, args):
        """
        선택된 모드에 따라 적절한 포즈 앱 인스턴스를 생성하고 실행합니다.
        """
        if self.mode == ModeNumber.SINGLE:
            pose_app = SinglePlayerApp(args, self.model, self.use_half)
            print("싱글 플레이어 모드 시작...")

        elif self.mode == ModeNumber.MULTIPLE:
            # TODO: 플레이어 수를 결정하는 로직을 추가해야 합니다.
            # 예: 사용자에게 플레이어 수를 입력받거나 기본값을 사용합니다.
            player_count = 2 # 기본 2명으로 가정
            pose_app = MultiPlayerApp(args, self.model, self.use_half, player_count=player_count)
            print("멀티 플레이어 모드 시작...")

        else:
            QMessageBox.warning(self, "오류", "알 수 없는 모드입니다.")
            return

        # QStackedWidget에 새 페이지를 추가하고 화면을 전환합니다.
        self.stack.addWidget(pose_app)
        self.stack.setCurrentWidget(pose_app)

        # 게임 종료 후 메인/랭킹 페이지로 돌아가는 시그널 연결
        pose_app.goMainRequested.connect(lambda: self.stack.setCurrentIndex(PageIndex.MAIN))
        pose_app.goRankRequested.connect(lambda: self.stack.setCurrentIndex(PageIndex.RANK))