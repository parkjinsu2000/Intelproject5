import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QListWidget, QListWidgetItem
)
from PyQt5.QtCore import Qt, pyqtSignal, QUrl
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from ..config import DirPath, FileName

class RankPage(QWidget):
    backRequested = pyqtSignal()

    def __init__(self):
        super().__init__()

        self.rank_file = os.path.join(DirPath.RANK_DIR, FileName.RANK_FILE)

        # 전체 레이아웃
        self.rank_main_verticalLayout = QVBoxLayout(self)

        # -------- 리스트 + 영상 영역 --------
        self.rank_HL = QHBoxLayout()

        # 왼쪽: 리스트 2개 (상단/하단)
        self.rank_list_VL = QVBoxLayout()
        self.video_list_listWidget = QListWidget()   # 좌측 상단 (영상 인기차트)
        self.rank_list_VL.addWidget(self.video_list_listWidget)
        self.player_video_listWidget = QListWidget() # 좌측 하단 (상세 기록)
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
        self.video_list_listWidget.itemClicked.connect(self.on_video_selected)

        # -------- 파일 처리 --------
        self.init_rank_db()
        self.load_ranking()

    # DB 디렉터리 초기화
    def init_rank_db(self):
        os.makedirs(DirPath.RANK_DIR, exist_ok=True)
        os.makedirs(DirPath.DETAILS_DIR, exist_ok=True)
        if not os.path.exists(self.rank_file):
            with open(self.rank_file, "w", encoding="utf-8") as f:
                f.write("")

    # 인기차트 로드
    def load_ranking(self):
        """rank_video_list.txt에서 영상 목록 불러오기 + 플레이 횟수 기준 정렬"""
        self.video_list_listWidget.clear()
        self.player_video_listWidget.clear()

        if not os.path.exists(self.rank_file):
            return

        videos = []
        with open(self.rank_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # 상세 기록 파일 확인
                details_file = os.path.join(DirPath.DETAILS_DIR, f"{line}.txt")
                if not os.path.exists(details_file):
                    with open(details_file, "w", encoding="utf-8") as df:
                        df.write("")

                # 플레이 횟수 = 줄 개수
                with open(details_file, "r", encoding="utf-8") as df:
                    play_count = sum(1 for _ in df)

                videos.append((line, play_count))

        # 플레이 횟수 내림차순 정렬
        videos.sort(key=lambda x: x[1], reverse=True)

        # 리스트 위젯에 추가
        for video_name, play_count in videos:
            item = QListWidgetItem(f"{video_name}  ({play_count}회)")
            item.setData(Qt.UserRole, video_name)  # 실제 이름 저장
            self.video_list_listWidget.addItem(item)

    # 영상 선택 시: 상세 기록 + 영상 재생
    def on_video_selected(self, item):
        video_name = item.data(Qt.UserRole)
        details_file = os.path.join(DirPath.DETAILS_DIR, f"{video_name}.txt")

        # 상세 기록 파일이 없으면 생성
        if not os.path.exists(details_file):
            with open(details_file, "w", encoding="utf-8") as f:
                f.write("")

        # 좌측 하단 리스트 갱신
        self.player_video_listWidget.clear()
        with open(details_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.player_video_listWidget.addItem(line)

        # 영상 재생
        video_path = os.path.join(DirPath.BASE_VIDEO_DIR, video_name)
        if not os.path.exists(video_path):
            print(f"[ERROR] 파일이 존재하지 않음: {video_path}")
            return
        url = QUrl.fromLocalFile(video_path)
        self.media_player.setMedia(QMediaContent(url))
        self.media_player.play()

    # rank_video_list.txt에 추가
    def file_write(self, video_name):
        """새 영상 이름 추가"""
        with open(self.rank_file, "a", encoding="utf-8") as f:
            f.write(f"{video_name}\n")
        self.load_ranking()

    def on_back_to_main(self):
        """메인으로 버튼 눌렀을 때"""
        self.media_player.stop()
        self.backRequested.emit()
