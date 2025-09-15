import os
from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton,
    QSpacerItem, QSizePolicy, QLineEdit
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPixmap
from .page_enum import ModeNumber

class MainPage(QWidget):
    viewRankRequested = pyqtSignal()
    challengeStartRequested = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.mode = ModeNumber.SINGLE

        # 레이아웃
        self.horizontalLayout = QHBoxLayout(self)

        # 왼쪽 이미지
        self.main_left_layout = QVBoxLayout()
        self.main_left_image_label = QLabel()
        self.main_left_image_label.setAlignment(Qt.AlignCenter)
        self.main_left_layout.addWidget(self.main_left_image_label)

        # 오른쪽 컨트롤
        self.main_right_layout = QVBoxLayout()
        self.main_right_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        self.select_challenge_single_pushButton = QPushButton("챌린지 선택 (싱글 플레이어)")
        self.main_right_layout.addWidget(self.select_challenge_single_pushButton)

        self.select_challenge_multiple_pushButton = QPushButton("챌린지 선택 (멀티 플레이어)")
        self.main_right_layout.addWidget(self.select_challenge_multiple_pushButton)

        self.view_rank_pushButton = QPushButton("랭킹 보기")
        self.main_right_layout.addWidget(self.view_rank_pushButton)

        self.ID_lineEdit = QLineEdit()
        self.ID_lineEdit.setPlaceholderText("아이디를 입력하세요")
        self.main_right_layout.addWidget(self.ID_lineEdit)

        self.Name_lineEdit = QLineEdit()
        self.Name_lineEdit.setPlaceholderText("이름을 입력하세요")
        self.main_right_layout.addWidget(self.Name_lineEdit)

        self.main_right_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # 합치기
        self.horizontalLayout.addLayout(self.main_left_layout, stretch=1)
        self.horizontalLayout.addLayout(self.main_right_layout, stretch=1)

        # 시그널 연결
        self.view_rank_pushButton.clicked.connect(self.viewRankRequested.emit)
        self.select_challenge_single_pushButton.clicked.connect(
            lambda: self.challengeStartRequested.emit(ModeNumber.SINGLE)
        )
        self.select_challenge_multiple_pushButton.clicked.connect(
            lambda: self.challengeStartRequested.emit(ModeNumber.MULTIPLE)
        )

        # 디렉터리 자동 생성
        img_dir = "resources/images"
        os.makedirs(img_dir, exist_ok=True)

        # 이미지
        self._orig_pix = QPixmap()
        self.set_image("resources/images/main_Image.png")

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