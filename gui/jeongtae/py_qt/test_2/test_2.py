# -*- coding: utf-8 -*-
import sys
import os
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QStackedWidget,
    QPushButton, QApplication, QLabel,
    QHBoxLayout, QVBoxLayout, QMessageBox, QListWidget,
    QSpacerItem, QSizePolicy, QLineEdit, QListWidgetItem
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, pyqtSignal, QUrl
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget

class MainPage(QWidget):
    viewRankRequested = pyqtSignal()
    challengeStartRequested = pyqtSignal()

    def __init__(self):
        super().__init__()

        # 전체 레이아웃
        self.horizontalLayout = QHBoxLayout(self)

        # -------- 왼쪽 (이미지 영역) --------
        self.main_left_layout = QVBoxLayout()
        self.main_left_image_label = QLabel()
        self.main_left_image_label.setAlignment(Qt.AlignCenter)
        self.main_left_layout.addWidget(self.main_left_image_label)

        # -------- 오른쪽 (컨트롤 영역) --------
        self.main_right_layout = QVBoxLayout()

        self.main_right_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        self.select_challenge_pushButton = QPushButton("챌린지 선택")
        self.main_right_layout.addWidget(self.select_challenge_pushButton)

        self.view_rank_pushButton = QPushButton("랭킹 보기")
        self.main_right_layout.addWidget(self.view_rank_pushButton)

        self.ID_lineEdit = QLineEdit()
        self.ID_lineEdit.setPlaceholderText("아이디를 입력하세요")
        self.main_right_layout.addWidget(self.ID_lineEdit)

        self.Name_lineEdit = QLineEdit()
        self.Name_lineEdit.setPlaceholderText("이름을 입력하세요")
        self.main_right_layout.addWidget(self.Name_lineEdit)

        self.main_right_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # -------- 레이아웃 합치기 --------
        self.horizontalLayout.addLayout(self.main_left_layout, stretch=1)
        self.horizontalLayout.addLayout(self.main_right_layout, stretch=1)

        # -------- 시그널 연결 --------
        self.view_rank_pushButton.clicked.connect(self.viewRankRequested.emit)
        self.select_challenge_pushButton.clicked.connect(self.challengeStartRequested.emit)

        # -------- 이미지 설정 --------
        self._orig_pix = QPixmap()
        self.set_image("main_Image.png")

    def set_image(self, path: str):
        pix = QPixmap(path)
        if pix.isNull():
            self.main_left_image_label.setText("이미지 로드 실패")
            self._orig_pix = QPixmap()
            return
        self._orig_pix = pix
        self._update_label_pixmap()

    def _update_label_pixmap(self):
        if self._orig_pix.isNull():
            return
        target = self._orig_pix.scaled(
            self.main_left_image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.main_left_image_label.setPixmap(target)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._update_label_pixmap()


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
        if not os.path.exists("rank_video_list.txt"):
            with open("rank_video_list.txt", "w", encoding="utf-8") as f:
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



class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.mainPage = MainPage()
        self.rankPage = RankPage()

        self.stack.addWidget(self.mainPage)  # index 0
        self.stack.addWidget(self.rankPage)  # index 1

        self.stack.setCurrentWidget(self.mainPage)
        
        self.init_ui()
        self.init_connect()

    def init_connect(self):
        # MainPage의 시그널을 받아서 스택 전환
        self.mainPage.viewRankRequested.connect(lambda: self.stack.setCurrentWidget(self.rankPage))
        # MainWindow.__init__ 안에서, 스택/페이지 추가 후에
        self.rankPage.backRequested.connect(lambda: self.stack.setCurrentWidget(self.mainPage))

        # (옵션) 챌린지 시작 시 하고 싶은 동작
        self.mainPage.challengeStartRequested.connect(self.on_challenge_start)

    def init_ui(self):
        self.resize(1280, 720)

    def on_challenge_start(self):
        QMessageBox.information(self, "Challenge", "챌린지를 시작합니다!")


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec_()


if __name__ == '__main__':
    main()
