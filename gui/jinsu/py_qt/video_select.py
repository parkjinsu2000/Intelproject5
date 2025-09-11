import os
from PyQt5.QtWidgets import (
    QWidget, QLabel, QListWidget, QPushButton,
    QVBoxLayout, QHBoxLayout, QMessageBox
)
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import Qt, QUrl, QFileInfo
from types import SimpleNamespace
from widget import PoseScoreApp
import torch

class VideoSelectPage(QWidget):
    def __init__(self, stacked_widget, model, use_half):
        super().__init__()
        self.stacked_widget = stacked_widget
        self.model = model
        self.use_half = use_half
        self.video_dir = "videos"
        self.ref_path = None

        # 왼쪽: 영상 선택 영역
        self.title_label = QLabel("🎬 영상선택")
        self.title_label.setAlignment(Qt.AlignCenter)

        self.video_list = QListWidget()
        self.video_list.itemClicked.connect(self.select_video)

        self.start_btn = QPushButton("시작")
        self.start_btn.clicked.connect(self.launch_pose_app)

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.title_label)
        left_layout.addWidget(self.video_list)
        left_layout.addWidget(self.start_btn)
        left_widget = QWidget()
        left_widget.setLayout(left_layout)

        # 오른쪽: 영상 재생 영역
        self.video_widget = QVideoWidget()
        self.player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.player.setVideoOutput(self.video_widget)
        self.player.setVolume(100)  # 오디오 볼륨 설정

        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel("🎥 영상재생"))
        right_layout.addWidget(self.video_widget)
        right_widget = QWidget()
        right_widget.setLayout(right_layout)

        # 전체 2열 레이아웃
        main_layout = QHBoxLayout()
        main_layout.addWidget(left_widget, 1)
        main_layout.addWidget(right_widget, 2)
        self.setLayout(main_layout)

        # 영상 목록 불러오기
        self.load_videos()

    def load_videos(self):
        if not os.path.exists(self.video_dir):
            os.makedirs(self.video_dir)

        for fname in os.listdir(self.video_dir):
            if fname.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                self.video_list.addItem(fname)

    def select_video(self, item):
        self.ref_path = os.path.join(self.video_dir, item.text())
        abs_path = QFileInfo(self.ref_path).absoluteFilePath()
        self.player.setMedia(QMediaContent(QUrl.fromLocalFile(abs_path)))
        self.player.play()

        print("[debug] 선택된 영상 경로:", self.ref_path)
        print("[debug] 절대 경로:", abs_path)
        print("[debug] 파일 존재 여부:", os.path.exists(self.ref_path))

    def launch_pose_app(self):
        if not self.ref_path:
            QMessageBox.warning(self, "선택 오류", "영상을 선택해주세요.")
            return

        args = SimpleNamespace(
            ref=self.ref_path,
            cam=0,
            start=3.0,
            every=10,
            no_mirror=False,
            disp_scale=1.0,
            disp_width=1280,
            save="output.mp4",
            imgsz=320,
            device="cuda" if torch.cuda.is_available() else "cpu",
            half=True,
            conf_thres=0.25
        )

        pose_app = PoseScoreApp(args, self.model, self.use_half)
        self.stacked_widget.addWidget(pose_app)
        self.stacked_widget.setCurrentWidget(pose_app)

    def resizeEvent(self, event):
        h = self.height()
        font_size = max(12, int(h / 30))
        self.video_list.setStyleSheet(f"font-size: {font_size}px;")
        self.title_label.setStyleSheet(f"font-size: {font_size + 6}px; font-weight: bold;")
        self.start_btn.setStyleSheet(f"font-size: {font_size + 2}px; padding: 10px;")
        super().resizeEvent(event)
