import cv2
import json
import numpy as np
import torch
import sys
import time
from PyQt5.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QSizePolicy,
    QStackedWidget, QSplitter, QMessageBox, QGraphicsOpacityEffect, QHBoxLayout
)
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import QTimer, Qt, QUrl, QFileInfo, QSize, QEvent, QRect, QPoint, QPropertyAnimation, QEasingCurve, QCoreApplication
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPainter, QBrush

from core.model_loader import make_infer
from core.pose_utils import (
    normalize_keypoints, pose_to_anglevec, frame_score_strict
)

from PyQt5.QtWidgets import QPushButton
from PyQt5.QtCore import pyqtSignal

class MyVideoWidget(QVideoWidget):
    """QVideoWidget을 상속받아 sizeHint를 오버라이드하여 레이아웃 내에서 유연하게 크기 조절"""
    def sizeHint(self):
        return QSize(1, 1)

class PoseScoreApp(QWidget):
    """
    포즈 교정 앱의 메인 위젯 클래스:
    왼쪽(참고 영상)과 오른쪽(웹캠) 화면을 QSplitter로 반반 분할하여 관리
    """
    goMainRequested = pyqtSignal()
    goRankRequested = pyqtSignal()

    def __init__(self, args, model, use_half):
        super().__init__()
        self.args = args
        self.model = model
        self.use_half = use_half
        self.cap_index = 0
        self.setMinimumSize(400, 300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.button_container = None  # 버튼들을 담을 위젯

        self.infer_pose = make_infer(self.model, self.args, self.use_half)
        print("Using pre-loaded pose detection model.")

        if self.args.json is None:
            error_message = "Error: JSON file path not provided. Please provide a path using the --json argument."
            print(error_message)
            QMessageBox.critical(self, "Error", error_message)
            self.close()
            return

        try:
            with open(self.args.json, 'r') as f:
                self.reference_data = json.load(f)["frames"]
                print(f"Successfully loaded {len(self.reference_data)} reference frames.")
        except FileNotFoundError:
            print(f"Error: Reference JSON file '{self.args.json}' not found.")
            self.reference_data = None
            QMessageBox.critical(self, "Error", f"Reference JSON file '{self.args.json}' not found.")
            self.close()
            return

        # --- 1. 왼쪽 화면 (미리보기/재생 전환) 설정 ---
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.preview_label.setScaledContents(True)
        self.preview_label.setMinimumSize(1, 1)

        self.video_widget = MyVideoWidget()
        self.video_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_widget.setMinimumSize(1, 1)

        self.player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.player.setVideoOutput(self.video_widget)
        self.player.setVolume(100)
        # 비디오 재생이 끝났을 때를 감지하는 시그널 연결
        self.player.stateChanged.connect(self.handle_video_state)

        self.video_stack = QStackedWidget()
        self.video_stack.setContentsMargins(0, 0, 0, 0)
        self.video_stack.addWidget(self.preview_label)
        self.video_stack.addWidget(self.video_widget)
        self.video_stack.setCurrentWidget(self.preview_label)

        left_container = QWidget()
        left_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)
        left_layout.addWidget(self.video_stack)

        # --- 2. 오른쪽 화면 (웹캠) 설정 ---
        self.cam_label = QLabel()
        self.cam_label.setAlignment(Qt.AlignCenter)
        self.cam_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.cam_label.setScaledContents(True)
        self.cam_label.setMinimumSize(1, 1)
        self.score_threshold = 0.5

        # 카운트다운 오버레이
        self.overlay_label = QLabel("", self.cam_label)
        self.overlay_label.setAlignment(Qt.AlignCenter)
        self.overlay_label.setStyleSheet("color: red; font-weight: bold; background-color: transparent;")
        self.overlay_label.setFont(QFont("Arial", int(96 * 1.5)))
        self.overlay_label.setAttribute(Qt.WA_TransparentForMouseEvents)

        # 피드백 텍스트를 위한 QLabel 추가
        self.feedback_label = QLabel(self.cam_label)
        # 피드백 라벨 위치를 중앙이 아닌 왼쪽 상단으로 변경
        self.feedback_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.feedback_label.setContentsMargins(20, 20, 0, 0) # 여백 추가
        self.feedback_label.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.feedback_label.hide() # 초반에는 숨김

        right_container = QWidget()
        right_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        right_layout.addWidget(self.cam_label)

        # --- 3. Splitter 및 전체 레이아웃 설정 ---
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(left_container)
        self.splitter.addWidget(right_container)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 1)

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.addWidget(self.splitter)

        if self.args.ref is not None:
            self.show_preview_frame(self.args.ref)

        self.cap = None
        self.init_webcam()

        self.count = 6  # 5초 카운트다운을 위해 6부터 시작
        self.count_timer = QTimer(self)
        self.count_timer.timeout.connect(self.update_countdown)
        self.count_timer.start(1000)

        self.feedback = ""
        self.feedback_opacity_effect = QGraphicsOpacityEffect(self.feedback_label)
        self.feedback_label.setGraphicsEffect(self.feedback_opacity_effect)
        self.fade_animation = QPropertyAnimation(self.feedback_opacity_effect, b"opacity")
        self.fade_animation.setDuration(1500) # 1.5초 동안 서서히 사라짐
        self.fade_animation.setStartValue(1.0)
        self.fade_animation.setEndValue(0.0)
        self.fade_animation.setEasingCurve(QEasingCurve.OutQuad)
        self.fade_animation.finished.connect(self.feedback_label.hide)

        self.score_timer = QTimer(self)
        # 1초에 3번 계산하도록 333ms로 설정
        self.score_timer.timeout.connect(self.calculate_score)

        # 최종 점수 관련 변수 추가
        self.final_score = 80
        self.game_over = False

        # 최근 3개 프레임 점수를 저장할 리스트 (1초에 3번 계산되므로)
        self.score_history = []
        self.score_history_length = 3

        # 따라하기 기능을 위한 지연 시간 (밀리초)
        self.follow_delay_ms = 200

        QTimer.singleShot(0, self.equalize_splitter)

    def init_webcam(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None

        backends = [cv2.CAP_DSHOW, cv2.CAP_V4L2, cv2.CAP_ANY]
        for backend in backends:
            temp_cap = cv2.VideoCapture(self.cap_index, backend)
            if temp_cap.isOpened():
                print(f"웹캠을 찾았고 {backend} 백엔드를 사용합니다.")
                self.cap = temp_cap
                break
            else:
                temp_cap.release()

        if self.cap and self.cap.isOpened():
            resolutions = [(1280, 720), (640, 480), (1920, 1080), (800, 600)]
            found_res = False
            for width, height in resolutions:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                ret, _ = self.cap.read()
                if ret:
                    print(f"웹캠 해상도를 {width}x{height}로 설정했습니다.")
                    found_res = True
                    break

            if not found_res:
                print("경고: 선호하는 해상도 설정에 실패했습니다. 기본 해상도를 사용합니다.")

            self.frame_timer = QTimer(self)
            self.frame_timer.timeout.connect(self.update_frame)
            self.frame_timer.start(30)
        else:
            print(f"Error: 웹캠을 찾을 수 없습니다.")
            self.cam_label.setText("웹캠을 찾을 수 없습니다.")
            self.cam_label.setStyleSheet("background-color: black; color: white; font-size: 20px;")
            self.frame_timer = None

    def equalize_splitter(self):
        w = max(2, self.splitter.width())
        self.splitter.setSizes([w // 2, w - (w // 2)])

    def show_preview_frame(self, video_path):
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            self.preview_label.setPixmap(
                pixmap.scaled(self.preview_label.size(),
                             Qt.KeepAspectRatio,
                             Qt.SmoothTransformation)
            )

    def update_frame(self):
        # When the game is over, this timer function will stop drawing frames
        # and instead call the function to display the final score.
        if self.game_over:
            self.display_final_score()
            return

        if self.cap and self.cap.isOpened():
            try:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    print("Error: 웹캠에서 프레임을 읽어올 수 없습니다. 재연결을 시도합니다.")
                    self.cam_label.setText("웹캠에서 프레임을 읽어올 수 없습니다. 재연결 중...")
                    self.init_webcam()
                    return

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame_rgb.shape

                # QImage를 생성하여 QPixmap으로 변환
                qt_image = QImage(frame_rgb.data, w, h, ch * w, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image)

                self.cam_label.setPixmap(
                    pixmap.scaled(self.cam_label.size(),
                                 Qt.KeepAspectRatio,
                                 Qt.SmoothTransformation)
                )

                # 피드백 라벨의 위치와 크기를 웹캠 라벨의 크기에 맞춰 좌측 상단에 설정
                # 텍스트가 잘리지 않도록 상자 크기를 훨씬 크게 키웠습니다.
                self.feedback_label.setGeometry(
                    10, 10, int(self.cam_label.width() / 1.5), int(self.cam_label.height() / 3)
                )
                self.feedback_label.setFont(QFont("Arial", int(self.cam_label.height() / 15), QFont.Bold))

                # 카운트다운 라벨이 보일 때만 위치 재설정 (중앙)
                if not self.overlay_label.isHidden():
                    self.overlay_label.setGeometry(self.cam_label.rect())
                    overlay_font_size = int(self.cam_label.height() / 5)
                    self.overlay_label.setFont(QFont("Arial", overlay_font_size, QFont.Bold))


            except Exception as e:
                print(f"Exception during frame processing: {e}")
                self.cam_label.setText(f"처리 중 오류 발생: {e}")
                self.cam_label.setStyleSheet("background-color: black; color: white; font-size: 20px;")
                if self.frame_timer:
                    self.frame_timer.stop()

    def calculate_score(self):
        if not self.cap or not self.cap.isOpened() or self.count > 0 or self.game_over:
            return

        ret, frame = self.cap.read()
        if not ret or frame is None:
            return

        cam_kps, cam_conf = self.infer_pose(frame)
        current_score = -1.0 # 기본값

        if cam_kps is not None and len(self.reference_data) > 0:
            # 비디오 현재 위치를 기반으로 정답 프레임 인덱스 계산
            # '따라하기' 기능을 위해 현재 시점보다 200ms 이전의 프레임을 사용합니다.
            delayed_position = self.player.position() - self.follow_delay_ms
            if delayed_position < 0:
                delayed_position = 0

            # `delayed_position`은 밀리초 단위이므로, 이를 초 단위로 변환 후 30fps를 곱합니다.
            ref_frame_index = int(delayed_position / 1000 * 30)

            # 정답 데이터는 10프레임마다 저장되어 있으므로, 인덱스를 10으로 나눕니다.
            ref_data_index = ref_frame_index // 10

            if ref_data_index < len(self.reference_data):
                ref_data = self.reference_data[ref_data_index]
                ref_kps = np.array(ref_data["kps"])

                cam_kps_norm = normalize_keypoints(cam_kps)
                ref_kps_norm = normalize_keypoints(ref_kps)

                vec_ref = pose_to_anglevec(ref_kps_norm)
                vec_live = pose_to_anglevec(cam_kps_norm)

                score, _, _ = frame_score_strict(vec_ref, vec_live)
                current_score = score

        # 최근 점수를 히스토리에 추가
        if current_score != -1.0:
            self.score_history.append(current_score)

        # 히스토리 점수가 3개 이상 쌓이면 평균을 계산하여 피드백 출력
        if len(self.score_history) >= self.score_history_length:
            smoothed_score = np.mean(self.score_history)

            new_feedback = ""
            if smoothed_score >= 80.0:
                new_feedback = "PERFECT"
            elif smoothed_score >= 50.0:
                new_feedback = "GOOD"
            else:
                new_feedback = "BAD"

            # 최종 점수 업데이트 로직: PERFECT는 +1, BAD는 -1
            if new_feedback == "PERFECT":
                self.final_score = min(100, self.final_score + 1)
            elif new_feedback == "BAD":
                self.final_score = max(0, self.final_score - 1)
            # GOOD일 때는 점수를 유지

            # 피드백이 새로 들어오면 애니메이션을 다시 시작합니다.
            self.feedback = new_feedback
            self.feedback_label.setText(new_feedback)
            if new_feedback == "PERFECT":
                self.feedback_label.setStyleSheet("color: lime; font-weight: bold;")
            elif new_feedback == "GOOD":
                self.feedback_label.setStyleSheet("color: yellow; font-weight: bold;")
            else: # BAD
                self.feedback_label.setStyleSheet("color: red; font-weight: bold;")

            # 애니메이션을 멈추고 Opacity를 1.0으로 리셋 후 다시 시작
            self.fade_animation.stop()
            self.feedback_label.show()
            self.feedback_opacity_effect.setOpacity(1.0)
            self.fade_animation.start()

            # 피드백을 출력한 후, 다음 1초를 위해 히스토리를 초기화합니다.
            self.score_history = []


    def update_countdown(self):
        self.count -= 1
        if self.count > 0:
            self.overlay_label.setText(str(self.count))
        elif self.count == 0:
            self.overlay_label.setText("START")
        else:
            self.overlay_label.hide()
            self.count_timer.stop()
            self.video_stack.setCurrentWidget(self.video_widget)
            QTimer.singleShot(0, self.equalize_splitter)
            self.play_video()
            # 포즈 스코어 계산 주기를 10프레임에 해당하는 333ms로 변경합니다.
            self.score_timer.start(333)

    def play_video(self):
        if self.args.ref:
            abs_path = QFileInfo(self.args.ref).absoluteFilePath()
            self.player.setMedia(QMediaContent(QUrl.fromLocalFile(abs_path)))
            self.player.play()

    def handle_video_state(self, state):
        """비디오 상태 변화를 감지하여 게임 종료를 처리합니다."""
        # 비디오가 멈췄을 때만 최종 점수를 표시
        if state == QMediaPlayer.StoppedState:
            print("비디오 재생이 종료되었습니다. 최종 점수를 표시합니다.")
            self.game_over = True
            # 비디오가 멈추면 모든 타이머를 즉시 중지합니다.
            if self.frame_timer:
                self.frame_timer.stop()
            if self.score_timer:
                self.score_timer.stop()
            if self.cap and self.cap.isOpened():
                self.cap.release()   # 웹캠 자원 즉시 해제
            self.display_final_score()

    def display_final_score(self):
        """최종 점수를 화면에 그립니다."""
        # Ensure the cam_label has a valid size before drawing
        label_size = self.cam_label.size()
        if label_size.width() <= 1 or label_size.height() <= 1:
            print("Warning: cam_label size is too small to draw final score. Retrying...")
            QTimer.singleShot(50, self.display_final_score)
            return

        pixmap = QPixmap(label_size)
        pixmap.fill(QColor(0, 0, 0)) # 검은 배경
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.TextAntialiasing)

        # 제목
        title_font = QFont("Arial", 40, QFont.Bold)
        painter.setFont(title_font)
        painter.setPen(QColor(255, 255, 255))
        title_rect = QRect(0, int(pixmap.height() * 0.2), pixmap.width(), 50)
        painter.drawText(title_rect, Qt.AlignCenter, "FINAL SCORE")

        # 점수
        score_font = QFont("Arial", 120, QFont.Bold)
        painter.setFont(score_font)

        # 점수 정수형 변환
        score_to_display = int(self.final_score)

        # 점수 색상 설정
        if score_to_display >= 80:
            painter.setPen(QColor(0, 255, 0))
        elif score_to_display >= 50:
            painter.setPen(QColor(255, 255, 0))
        else:
            painter.setPen(QColor(255, 0, 0))

        score_rect = QRect(0, int(pixmap.height() * 0.4), pixmap.width(), 200)
        painter.drawText(score_rect, Qt.AlignCenter, str(score_to_display))

        painter.end()
        self.cam_label.setPixmap(pixmap)

        # --- 버튼 UI 추가 ---
        if self.button_container is None:
            self.button_container = QWidget(self.cam_label)
            layout = QHBoxLayout(self.button_container)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(20)

            self.main_btn = QPushButton("메인으로")
            self.rank_btn = QPushButton("랭킹 보기")

            font = QFont("Arial", 20, QFont.Bold)
            self.main_btn.setFont(font)
            self.rank_btn.setFont(font)

            self.main_btn.setStyleSheet("background-color: white; padding: 10px;")
            self.rank_btn.setStyleSheet("background-color: white; padding: 10px;")

            layout.addStretch(1)
            layout.addWidget(self.main_btn)
            layout.addWidget(self.rank_btn)
            layout.addStretch(1)

            self.button_container.setLayout(layout)
            self.button_container.setGeometry(
                0, int(self.cam_label.height() * 0.8), 
                self.cam_label.width(), 60
            )
            self.button_container.show()

            # 시그널 연결
            self.main_btn.clicked.connect(self.goMainRequested.emit)
            self.rank_btn.clicked.connect(self.goRankRequested.emit)

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QEvent.WindowStateChange:
            QTimer.singleShot(0, self.equalize_splitter)

    def resizeEvent(self, event):
        QTimer.singleShot(0, self.equalize_splitter)
        super().resizeEvent(event)
        if self.count >= 0:
            self.overlay_label.setGeometry(self.cam_label.rect())
            overlay_font_size = int(self.cam_label.height() / 5)
            self.overlay_label.setFont(QFont("Arial", overlay_font_size, QFont.Bold))
        if self.game_over:
            self.display_final_score()
            if self.button_container:
                self.button_container.setGeometry(
                    0, int(self.cam_label.height() * 0.8),
                    self.cam_label.width(), 60
                )

    def closeEvent(self, event):
        if self.frame_timer:
            self.frame_timer.stop()
        if self.score_timer:
            self.score_timer.stop()
        if self.cap and self.cap.isOpened():
            self.cap.release()
        super().closeEvent(event)