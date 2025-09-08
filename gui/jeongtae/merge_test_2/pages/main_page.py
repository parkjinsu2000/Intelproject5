import os
from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton,
    QSpacerItem, QSizePolicy, QLineEdit
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPixmap

class MainPage(QWidget):
    viewRankRequested = pyqtSignal()
    challengeStartRequested = pyqtSignal()

    def __init__(self):
        super().__init__()

        layout = QHBoxLayout(self)

        # 왼쪽: 이미지
        self.left_layout = QVBoxLayout()
        self.image_label = QLabel("메인 이미지")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.left_layout.addWidget(self.image_label)

        # 오른쪽: 버튼들
        self.right_layout = QVBoxLayout()
        self.right_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        self.challenge_btn = QPushButton("챌린지 선택")
        self.right_layout.addWidget(self.challenge_btn)

        self.rank_btn = QPushButton("랭킹 보기")
        self.right_layout.addWidget(self.rank_btn)

        self.id_input = QLineEdit()
        self.id_input.setPlaceholderText("아이디 입력")
        self.right_layout.addWidget(self.id_input)

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("이름 입력")
        self.right_layout.addWidget(self.name_input)

        self.right_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        layout.addLayout(self.left_layout, 1)
        layout.addLayout(self.right_layout, 1)

        # 시그널 연결
        self.rank_btn.clicked.connect(self.viewRankRequested.emit)
        self.challenge_btn.clicked.connect(self.challengeStartRequested.emit)

        # 이미지 폴더 준비
        img_dir = "resources/images"
        os.makedirs(img_dir, exist_ok=True)
        self.set_image(os.path.join(img_dir, "main_Image.png"))

    def set_image(self, path: str):
        pix = QPixmap(path)
        if pix.isNull():
            self.image_label.setText("이미지 없음")
            return
        self.image_label.setPixmap(pix.scaled(
            self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.set_image("resources/images/main_Image.png")
