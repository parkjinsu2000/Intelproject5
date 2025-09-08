from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QListWidget
)
from PyQt5.QtCore import pyqtSignal

class RankPage(QWidget):
    backRequested = pyqtSignal()

    def __init__(self):
        super().__init__()

        layout = QVBoxLayout(self)

        # 리스트 2개
        hlayout = QHBoxLayout()
        self.video_list = QListWidget()
        self.player_list = QListWidget()
        hlayout.addWidget(self.video_list, 1)
        hlayout.addWidget(self.player_list, 1)

        layout.addLayout(hlayout, 9)

        # 하단 버튼
        self.back_btn = QPushButton("메인으로")
        self.back_btn.clicked.connect(self.backRequested.emit)
        layout.addWidget(self.back_btn, 1)
