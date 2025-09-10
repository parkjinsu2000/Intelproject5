import os
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QLabel, QSplitter, QPushButton
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import Qt, QUrl, pyqtSignal
from config import DirPath

class UserVideoPage(QWidget):
    backRequested = pyqtSignal()

    def __init__(self):
        super().__init__()

        main_layout = QVBoxLayout(self)

        self.splitter = QSplitter(Qt.Horizontal)

        # 좌측 레이아웃
        self.left_layout = QVBoxLayout()
        self.user_list = QListWidget()
        self.multi_list = QListWidget()
        self.left_layout.addWidget(QLabel("싱글 모드 녹화 영상"))
        self.left_layout.addWidget(self.user_list)
        self.left_layout.addWidget(QLabel("멀티 모드 녹화 영상"))
        self.left_layout.addWidget(self.multi_list)

        self.left_widget = QWidget()
        self.left_widget.setLayout(self.left_layout)

        # 우측 영상 플레이어
        self.video_widget = QVideoWidget()
        self.player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.player.setVideoOutput(self.video_widget)

        # Splitter 구성
        self.splitter.addWidget(self.left_widget)
        self.splitter.addWidget(self.video_widget)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 2)

        main_layout.addWidget(self.splitter)

        # -------- 하단 컨트롤 --------
        self.control_layout = QHBoxLayout()
        self.back_btn = QPushButton("뒤로가기")
        self.back_btn.clicked.connect(self.backRequested.emit)
        self.control_layout.addStretch(1)
        self.control_layout.addWidget(self.back_btn)
        self.control_layout.addStretch(1)

        main_layout.addLayout(self.control_layout)

        # 리스트 이벤트 연결
        self.user_list.itemClicked.connect(lambda item: self.play_video(DirPath.USER_VIDEO_DIR, item.text()))
        self.multi_list.itemClicked.connect(lambda item: self.play_video(DirPath.USER_MULTI_VIDEO_DIR, item.text()))

        self.load_videos()

    def load_videos(self):
        """유저/멀티 영상 디렉터리 생성 후 리스트 갱신"""
        os.makedirs(DirPath.USER_VIDEO_DIR, exist_ok=True)
        os.makedirs(DirPath.USER_MULTI_VIDEO_DIR, exist_ok=True)

        self.user_list.clear()
        self.multi_list.clear()

        for fname in sorted(os.listdir(DirPath.USER_VIDEO_DIR)):
            if fname.lower().endswith(".mp4"):
                self.user_list.addItem(fname)

        for fname in sorted(os.listdir(DirPath.USER_MULTI_VIDEO_DIR)):
            if fname.lower().endswith(".mp4"):
                self.multi_list.addItem(fname)

    def play_video(self, base_dir, fname):
        path = os.path.join(base_dir, fname)
        if not os.path.exists(path):
            print(f"[WARN] 파일이 존재하지 않음: {path}")
            return
        self.player.setMedia(QMediaContent(QUrl.fromLocalFile(path)))
        self.player.play()
