import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QListWidget, QListWidgetItem
)
from PyQt5.QtCore import Qt, pyqtSignal, QUrl
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget

class RankPage(QWidget):
    backRequested = pyqtSignal()

    BASE_VIDEO_DIR = os.path.abspath(
        "/home/ubuntu/workspace_intel/Intelproject5/gui/jeongtae/py_qt/test_2/"
    )  # 원하는 기본 절대경로

    def __init__(self):
        super().__init__()

        # 전체 레이아웃
        self.rank_main_verticalLayout = QVBoxLayout(self)

        # -------- 리스트 + 영상 영역 --------
        self.rank_HL = QHBoxLayout()

        # 왼쪽: 리스트 2개
        self.rank_list_VL = QVBoxLayout()
        self.video_list_listWidget = QListWidget()
        self.rank_list_VL.addWidget(self.video_list_listWidget)
        self.player_video_listWidget = QListWidget()
        self.rank_list_VL.addWidget(self.player_video_listWidget)

        # 오른쪽: 영상 플레이 영역
        self.video_play_VL = QVBoxLayout()
        self.video_widget = QVideoWidget()
        self.video_play_VL.addWidget(self.video_widget)

        self.rank_HL.addLayout(self.rank_list_VL, stretch=5)
        self.rank_HL.addLayout(self.video_play_VL, stretch=5)
        self.rank_main_verticalLayout.addLayout(self.rank_HL, stretch=9)

        # -------- 하단 컨트롤 --------
        self.rank_control_HL = QHBoxLayout()
        self.go_to_main_PB = QPushButton("메인으로")
        self.rank_control_HL.addWidget(self.go_to_main_PB)
        self.rank_main_verticalLayout.addLayout(self.rank_control_HL, stretch=1)

        # -------- 미디어 플레이어 --------
        self.media_player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.media_player.setVideoOutput(self.video_widget)

        # -------- 시그널 연결 --------
        self.go_to_main_PB.clicked.connect(self.on_back_to_main)
        self.video_list_listWidget.itemClicked.connect(self.play_selected_video)

        # -------- 파일 처리 --------
        self.init_rank_db()
        self.load_ranking()

    def init_rank_db(self):
        os.makedirs(os.path.dirname("resources/"), exist_ok=True)
        if not os.path.exists("resources/rank_video_list.txt"):
            with open("resources/rank_video_list.txt", "w", encoding="utf-8") as f:
                f.write("")

    def load_ranking(self):
        """파일에서 영상 리스트를 불러와 ListWidget에 추가 (표시는 파일명만)"""
        self.video_list_listWidget.clear()
        self.player_video_listWidget.clear()
        if os.path.exists("rank_video_list.txt"):
            with open("rank_video_list.txt", "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        abs_path = os.path.join(self.BASE_VIDEO_DIR, line)
                        filename = os.path.basename(line)

                        # 리스트에는 파일명만 표시, 절대경로는 data에 저장
                        item = QListWidgetItem(filename)
                        item.setData(Qt.UserRole, abs_path)
                        self.video_list_listWidget.addItem(item)

    def file_write(self, msg):
        """rank_video_list.txt에 파일명만 저장"""
        with open("rank_video_list.txt", "a", encoding="utf-8") as f:
            f.write(f"{msg}\n")
        self.load_ranking()

    # -------- 영상 재생 기능 --------
    def play_selected_video(self, item):
        video_path = item.data(Qt.UserRole)  # 절대경로 꺼내기
        if not os.path.exists(video_path):
            print(f"[ERROR] 파일이 존재하지 않음: {video_path}")
            return
        url = QUrl.fromLocalFile(video_path)
        self.media_player.setMedia(QMediaContent(url))
        self.media_player.play()

    def on_back_to_main(self):
        """메인으로 버튼 눌렀을 때"""
        self.media_player.stop()   # 영상 정지
        self.backRequested.emit()
