import os
from PyQt5.QtWidgets import QMainWindow, QStackedWidget, QMessageBox
from .main_page import MainPage
from .rank_page import RankPage
from .video_select_page import VideoSelectPage
from .enums import PageIndex, ModeNumber
# from .pose_score_app import PoseScoreApp
from .Single_Player_app import SinglePlayerApp
from .Multi_Player_app import MultiPlayerApp
from .user_video_page import UserVideoPage

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
        self.userVideoPage = UserVideoPage()

        self.stack.addWidget(self.mainPage)                     # PageIndex.MAIN
        self.stack.addWidget(self.rankPage)                     # PageIndex.RANK
        self.stack.addWidget(self.videoPage)                    # PageIndex.VIDEO_SELECT
        self.stack.addWidget(self.userVideoPage)                # PageIndex.USER_VIDEO_PAGE

        self.stack.setCurrentIndex(PageIndex.MAIN)

        self.signal_connect()

        self.resize(1280, 720)

    def signal_connect(self):
        # 시그널 연결
        self.mainPage.viewRankRequested.connect(
            lambda: self.stack.setCurrentIndex(PageIndex.RANK)
        )
        self.rankPage.backRequested.connect(
            lambda: self.stack.setCurrentIndex(PageIndex.MAIN)
        )
        self.mainPage.challengeStartRequested.connect(self.on_challenge_start)
        self.mainPage.newChallengeVideoAdded.connect(self.videoPage.load_videos)
        self.mainPage.newChallengeVideoAdded.connect(self.rankPage.load_ranking)

        self.mainPage.viewUserVideoRequested.connect(
            lambda: self.stack.setCurrentIndex(PageIndex.USER_VIDEO_PAGE)
        )
        self.videoPage.startPoseAppRequested.connect(self.launch_pose_app)
        
        self.userVideoPage.backRequested.connect(
            lambda: self.stack.setCurrentIndex(PageIndex.MAIN)
        )

    def on_challenge_start(self, mode: ModeNumber):
        """챌린지 선택 버튼 → VideoSelectPage"""
        self.mode = mode
        self.mainPage.mode = self.mode
        self.stack.setCurrentIndex(PageIndex.VIDEO_SELECT)

    def launch_pose_app(self, args):
        """
        선택된 모드에 따라 적절한 포즈 앱 인스턴스를 생성하고 실행합니다.
        """
        user_id = self.mainPage.ID_lineEdit.text().strip()
        user_name = self.mainPage.Name_lineEdit.text().strip()

        if not user_id or not user_name:
            QMessageBox.warning(self, "입력 오류", "아이디와 이름을 모두 입력해야 합니다.")
            return

        # VideoSelectPage 에서 ref_path 로 선택된 영상 파일명 추출
        video_title = os.path.splitext(os.path.basename(self.videoPage.ref_path))[0] \
                    if self.videoPage.ref_path else "untitled"

        if self.mode == ModeNumber.SINGLE:
            pose_app = SinglePlayerApp(
                args, self.model, self.use_half,
                user_name=user_name,
                user_id=user_id,
                video_title=video_title
            )
            print("싱글 플레이어 모드 시작...")

        elif self.mode == ModeNumber.MULTIPLE:
            pose_app = MultiPlayerApp(
                args, self.model, self.use_half,
                user_name=user_name,
                user_id=user_id,
                video_title=video_title
            )
            print("멀티 플레이어 모드 시작...")

        else:
            QMessageBox.warning(self, "오류", "알 수 없는 모드입니다.")
            return

        self.stack.addWidget(pose_app)
        self.stack.setCurrentWidget(pose_app)

        pose_app.goMainRequested.connect(lambda: self.stack.setCurrentIndex(PageIndex.MAIN))
        pose_app.goRankRequested.connect(lambda: self.stack.setCurrentIndex(PageIndex.RANK))
