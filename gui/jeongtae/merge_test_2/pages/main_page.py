import os
import shutil
from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton,
    QSpacerItem, QSizePolicy, QLineEdit, QFileDialog, QMessageBox
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPixmap
from .enums import ModeNumber
from config import DirPath, FileName

class MainPage(QWidget):
    viewRankRequested = pyqtSignal()
    challengeStartRequested = pyqtSignal(int)
    newChallengeVideoAdded = pyqtSignal()

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

        self.add_new_challenge_video_pushButton = QPushButton("새 챌린지 영상 등록")
        self.main_right_layout.addWidget(self.add_new_challenge_video_pushButton)

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
        self.add_new_challenge_video_pushButton.clicked.connect(self.add_new_challenge_video)
        self.select_challenge_single_pushButton.clicked.connect(
            lambda: self.challengeStartRequested.emit(ModeNumber.SINGLE)
        )
        self.select_challenge_multiple_pushButton.clicked.connect(
            lambda: self.challengeStartRequested.emit(ModeNumber.MULTIPLE)
        )

        # 디렉터리 자동 생성
        os.makedirs(DirPath.IMAGE_DIR, exist_ok=True)
        os.makedirs(DirPath.REF_VIDEO_DIR, exist_ok=True)

        # 이미지
        self._orig_pix = QPixmap()
        self.set_image(os.path.join(DirPath.IMAGE_DIR, FileName.MAIN_IMAGE))

    def add_new_challenge_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "새 챌린지 영상 선택", "", "Video Files (*.mp4 *.avi *.mov *.mkv)"
        )
        if not file_path:
            return  # 사용자가 취소

        try:
            # 선택한 파일 이름만 추출
            filename = os.path.basename(file_path)
            dest_path = os.path.join(DirPath.REF_VIDEO_DIR, filename)

            # 디렉터리 보장
            os.makedirs(DirPath.REF_VIDEO_DIR, exist_ok=True)
            os.makedirs(DirPath.RANK_DIR, exist_ok=True)

            # 파일 복사 (덮어쓰기 방지)
            if not os.path.exists(dest_path):
                shutil.copy(file_path, dest_path)

            # 랭킹 파일 업데이트 (중복 방지)
            rank_file = os.path.join(DirPath.RANK_DIR, FileName.RANK_FILE)
            existing = set()
            if os.path.exists(rank_file):
                with open(rank_file, "r", encoding="utf-8") as f:
                    existing = {line.strip() for line in f}

            if filename not in existing:
                with open(rank_file, "a", encoding="utf-8") as f:
                    f.write(f"{filename}\n")

            self.newChallengeVideoAdded.emit()

            QMessageBox.information(
                self,
                "등록 완료",
                f"새 챌린지 영상이 등록되었습니다:\n{dest_path}"
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "오류",
                f"영상 등록 중 오류 발생:\n{e}"
            )


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
