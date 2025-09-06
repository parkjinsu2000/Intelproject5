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
from widget import PoseScoreApp
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
        # 필요시 라벨 유지. 비율 영향 줄이려면 폰트만 키우고 여백 0으로.
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
        self.splitter.setStretchFactor(1, 2)  # 선택 영역:재생 영역 = 1:2 기본
        # 초기 반반(또는 1:2) 맞추기
        QTimer.singleShot(0, self.equalize_splitter)

        root_layout = QHBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        root_layout.addWidget(self.splitter)

        self.load_videos()

    def equalize_splitter(self):
        # 현재 가용 폭을 기준으로 비율 고정(여기선 1:2)
        w = max(3, self.splitter.width())
        left = w // 3
        right = w - left
        self.splitter.setSizes([left, right])

        # 미리보기 비디오 위젯도 강제 갱신
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
        abs_path = QFileInfo(self.ref_path).absoluteFilePath()
        self.player.setMedia(QMediaContent(QUrl.fromLocalFile(abs_path)))
        self.player.play()

    def launch_pose_app(self):
        if not self.ref_path:
            QMessageBox.warning(self, "선택 오류", "영상을 선택해주세요.")
            return

        # ✅ 기존 영상 재생 중지
        self.player.stop()

        args = SimpleNamespace(
            ref=self.ref_path,
            cam=0,
            start=3.0,
            every=5,
            no_mirror=False,
            disp_scale=1.0,
            disp_width=1280,
            save="output.mp4",
            imgsz=320,
            device="cuda" if torch.cuda.is_available() else "cpu",
            half=self.use_half,
            conf_thres=0.25
        )

        pose_app = PoseScoreApp(args, self.model, self.use_half)
        # 페이지 자체도 Expanding 보장
        pose_app.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        if pose_app.layout():
            pose_app.layout().setContentsMargins(0, 0, 0, 0)
            pose_app.layout().setSpacing(0)

        self.stacked_widget.addWidget(pose_app)
        self.stacked_widget.setCurrentWidget(pose_app)

        # 전환 직후 한 틱 뒤에 비율/크기 강제 조정
        def nudge():
            if hasattr(pose_app, "equalize_splitter"):
                pose_app.equalize_splitter()
            elif hasattr(pose_app, "force_resize_video"):
                pose_app.force_resize_video()
        QTimer.singleShot(0, nudge)

    def resizeEvent(self, event):
        # 창 크기 변경 시마다 비율 재강제
        self.equalize_splitter()

        # 글꼴/패딩 반응형
        h = self.height()
        font_size = max(12, int(h / 30))
        self.video_list.setStyleSheet(f"font-size: {font_size}px;")
        self.title_label.setStyleSheet(f"font-size: {font_size + 6}px; font-weight: bold;")
        self.start_btn.setStyleSheet(f"font-size: {font_size + 2}px; padding: 10px;")
        super().resizeEvent(event)
