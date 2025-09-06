import os
from PyQt5.QtWidgets import (
    QWidget, QLabel, QListWidget, QPushButton,
    QVBoxLayout, QHBoxLayout, QMessageBox, QSizePolicy, QSplitter
)
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import Qt, QUrl, QFileInfo, QTimer, QSize
from PyQt5.QtGui import QFont
from types import SimpleNamespace
from pose_score_app import PoseScoreApp
from model_loader import make_infer
import torch

# QVideoWidget 상속 → sizeHint 무시해 레이아웃 비율에 영향 못 주게
class MyVideoWidget(QVideoWidget):
    def sizeHint(self):
        return QSize(0, 0)

class VideoSelectPage(QWidget):
    def __init__(self, stacked_widget, model, use_half):
        super().__init__()
        self.stacked_widget = stacked_widget
        self.model = model
        self.use_half = use_half
        self.video_dir = "videos"
        self.ref_path = None
        self.json_path = None

        self.setContentsMargins(0, 0, 0, 0)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # 왼쪽: 영상 선택 영역
        self.title_label = QLabel("🎬 영상선택")
        self.title_label.setAlignment(Qt.AlignCenter)

        self.video_list = QListWidget()
        self.video_list.itemClicked.connect(self.select_video)

        self.start_btn = QPushButton("시작")
        self.start_btn.clicked.connect(self.launch_pose_app)

        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)
        left_layout.addWidget(self.title_label)
        left_layout.addWidget(self.video_list)
        left_layout.addWidget(self.start_btn)

        left_widget = QWidget()
        left_widget.setLayout(left_layout)
        left_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # 오른쪽: 영상 재생 영역 (MyVideoWidget 사용)
        self.video_widget = MyVideoWidget()
        self.video_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_widget.setMinimumSize(0, 0)

        self.player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.player.setVideoOutput(self.video_widget)
        self.player.setVolume(100)

        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        header = QLabel("🎥 영상재생")
        header.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(header)
        right_layout.addWidget(self.video_widget)

        right_widget = QWidget()
        right_widget.setLayout(right_layout)
        right_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # 전체 레이아웃: QSplitter로 반반 강제
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(left_widget)
        self.splitter.addWidget(right_widget)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 2)
        QTimer.singleShot(0, self.equalize_splitter)

        root_layout = QHBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        root_layout.addWidget(self.splitter)

        self.load_videos()

    def equalize_splitter(self):
        w = max(3, self.splitter.width())
        left = w // 3
        right = w - left
        self.splitter.setSizes([left, right])
        self.video_widget.setMinimumSize(0, 0)
        self.video_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_widget.updateGeometry()

    def load_videos(self):
        os.makedirs(self.video_dir, exist_ok=True)
        self.video_list.clear()
        for fname in sorted(os.listdir(self.video_dir)):
            if fname.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                self.video_list.addItem(fname)

    def select_video(self, item):
        self.ref_path = os.path.join(self.video_dir, item.text())
        video_name, _ = os.path.splitext(item.text())
        json_fname = video_name + ".json"
        json_full_path = os.path.join(self.video_dir, json_fname)

        if os.path.exists(json_full_path):
            self.json_path = json_full_path
            print(f"JSON 파일이 감지되었습니다: {self.json_path}")
        else:
            self.json_path = None
            print(f"JSON 파일을 찾을 수 없습니다: {json_full_path}")

        abs_path = QFileInfo(self.ref_path).absoluteFilePath()
        self.player.setMedia(QMediaContent(QUrl.fromLocalFile(abs_path)))
        self.player.play()

    def launch_pose_app(self):
        if not self.ref_path:
            QMessageBox.warning(self, "선택 오류", "영상을 선택해주세요.")
            return

        self.player.stop()

        args = SimpleNamespace(
            ref=self.ref_path,
            json=self.json_path,
            cam=0,
            every=5,
            no_mirror=False,
            disp_scale=1.0,
            disp_width=1280,
            save="output.mp4",
            imgsz=320,
            device="cuda" if torch.cuda.is_available() else "cpu",
            half=self.use_half,
            conf_thres=0.25,
            step=5
        )

        pose_app = PoseScoreApp(args, self.model, self.use_half)
        pose_app.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        if pose_app.layout():
            pose_app.layout().setContentsMargins(0, 0, 0, 0)
            pose_app.layout().setSpacing(0)

        self.stacked_widget.addWidget(pose_app)
        self.stacked_widget.setCurrentWidget(pose_app)

        def nudge():
            if hasattr(pose_app, "equalize_splitter"):
                pose_app.equalize_splitter()
            elif hasattr(pose_app, "force_resize_video"):
                pose_app.force_resize_video()
        QTimer.singleShot(0, nudge)

    def resizeEvent(self, event):
        self.equalize_splitter()
        h = self.height()
        font_size = max(12, int(h / 30))
        self.video_list.setStyleSheet(f"font-size: {font_size}px;")
        self.title_label.setStyleSheet(f"font-size: {font_size + 6}px; font-weight: bold;")
        self.start_btn.setStyleSheet(f"font-size: {font_size + 2}px; padding: 10px;")
        super().resizeEvent(event)
