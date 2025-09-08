# -*- coding: utf-8 -*-
import sys
import os
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QStackedWidget,
    QPushButton, QApplication, QLabel,
    QHBoxLayout, QVBoxLayout, QMessageBox, QListWidget
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, pyqtSignal



class MainPage(QWidget):
    viewRankRequested = pyqtSignal()
    challengeStartRequested = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.mainPage_layout_h = QHBoxLayout()

        self.mainPage_left_panel = QWidget()
        self.right_panel = QWidget()

        self.main_left_image_layout_v = QVBoxLayout(self.mainPage_left_panel)
        self.main_right_control_layout_v = QVBoxLayout(self.right_panel)

        self.main_left_image_label = QLabel("이미지를 불러오는 중...")
        self.main_left_image_label.setAlignment(Qt.AlignCenter)  # 가운데 정렬

        self.main_right_control_challengeStart_btn = QPushButton("챌린지 선택")
        self.main_right_control_viewRank_btn = QPushButton("랭킹보기")

        self._orig_pix = QPixmap()  # ★ 원본 픽스맵 보관용

        self.init_ui()
        self.init_btn_connect()

        # ★ 원하는 PNG 경로로 교체
        self.set_image("main_Image.png")
        # 리소스 파일을 쓴다면: self.set_image(":/images/logo.png")

    def init_ui(self):
        self.mainPage_layout_h.addWidget(self.mainPage_left_panel)
        self.mainPage_layout_h.addWidget(self.right_panel)

        self.main_left_image_layout_v.addWidget(self.main_left_image_label)

        self.main_right_control_layout_v.addWidget(self.main_right_control_challengeStart_btn, alignment=Qt.AlignHCenter)
        self.main_right_control_layout_v.addWidget(self.main_right_control_viewRank_btn, alignment=Qt.AlignHCenter)

        self.mainPage_layout_h.setStretch(0, 2)
        self.mainPage_layout_h.setStretch(1, 1)

        self.setLayout(self.mainPage_layout_h)

    def init_btn_connect(self):
        self.main_right_control_viewRank_btn.clicked.connect(lambda: self.viewRankRequested.emit())
        self.main_right_control_challengeStart_btn.clicked.connect(lambda: self.challengeStartRequested.emit())

    # ---------- 이미지 관련 유틸 ----------
    def set_image(self, path: str):
        """PNG 파일을 로드해서 라벨에 표시(비율 유지, 선명 스케일)."""
        pix = QPixmap(path)
        if pix.isNull():
            self.main_left_image_label.setText("이미지 로드 실패")
            self._orig_pix = QPixmap()
            return
        self._orig_pix = pix
        self._update_label_pixmap()

    def _update_label_pixmap(self):
        """라벨 크기에 맞춰 비율 유지로 스케일 후 표시."""
        if self._orig_pix.isNull():
            return
        target = self._orig_pix.scaled(
            self.main_left_image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.main_left_image_label.setPixmap(target)

    def resizeEvent(self, e):
        """창/레이아웃 변경 시 이미지도 다시 스케일."""
        super().resizeEvent(e)
        self._update_label_pixmap()


class RankPage(QWidget):
    backRequested = pyqtSignal()
    
    def __init__(self):
        super().__init__()

        # 바깥쪽 레이아웃
        self.layout = QVBoxLayout(self)

        # 제목 라벨
        self.main_test_label = QLabel("랭킹 페이지", alignment=Qt.AlignHCenter)

        # 랭킹 표시용 리스트
        self.rank_list = QListWidget()

        # 뒤로가기 버튼
        self.back_btn = QPushButton("← 메인으로")
        self.back_btn.clicked.connect(self.backRequested.emit)

        # 위젯 배치
        self.layout.addWidget(self.main_test_label)
        self.layout.addWidget(self.rank_list)
        self.layout.addWidget(self.back_btn, alignment=Qt.AlignHCenter)

        # 파일 초기화 + 읽기
        self.init_rank_db()
        self.load_ranking()

    def init_rank_db(self):
        """rank_video_list.txt 없으면 생성"""
        if not os.path.exists("rank_video_list.txt"):
            with open("rank_video_list.txt", "w", encoding="utf-8") as f:
                f.write("")  # 빈 파일 생성

    def load_ranking(self):
        """파일 내용을 리스트 위젯에 표시"""
        self.rank_list.clear()  # 기존 내용 지우기
        if os.path.exists("rank_video_list.txt"):
            with open("rank_video_list.txt", "r", encoding="utf-8") as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if line:  # 빈 줄은 제외
                        self.rank_list.addItem(line)

    def file_write(self, msg):
        """파일에 한 줄 추가"""
        with open("rank_video_list.txt", "a", encoding="utf-8") as f:
            f.write(f"{msg}\n")
        self.load_ranking()  # 쓰고 나서 갱신


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
