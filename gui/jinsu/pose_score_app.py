import cv2
import json
import numpy as np
import torch
import sys
import time
from PyQt5.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QSizePolicy,
    QStackedWidget, QSplitter, QMessageBox
)
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import QTimer, Qt, QUrl, QFileInfo, QSize, QEvent, QRect, QPoint, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPainter, QBrush
from model_loader import make_infer
from pose_utils import (
    normalize_keypoints, pose_to_anglevec, frame_score_strict
)

class MyVideoWidget(QVideoWidget):
    """QVideoWidget을 상속받아 sizeHint를 오버라이드하여 레이아웃 내에서 유연하게 크기 조절"""
    def sizeHint(self):
        return QSize(1, 1)

class PoseScoreApp(QWidget):
    """
    포즈 교정 앱의 메인 위젯 클래스:
    왼쪽(참고 영상)과 오른쪽(웹캠) 화면을 QSplitter로 반반 분할하여 관리
    """
    def __init__(self, args, model, use_half):
        super().__init__()
        self.args = args
        self.model = model
        self.use_half = use_half
        self.cap_index = 0
        self.setMinimumSize(400, 300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.infer_pose = make_infer(self.model, self.args, self.use_half)
        print("Using pre-loaded pose detection model.")

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
        root_layout.setSpacing(0)
        root_layout.addWidget(self.splitter)

        self.show_preview_frame(self.args.ref)

        self.cap = None
        self.init_webcam()

        self.count = 4
        self.count_timer = QTimer(self)
        self.count_timer.timeout.connect(self.update_countdown)
        self.count_timer.start(1000)

        self.feedback = ""
        self.current_opacity = 1.0
        self.fade_step = 0.05
        self.fade_timer = QTimer(self)
        self.fade_timer.timeout.connect(self.update_fade)

        self.score_timer = QTimer(self)
        self.score_timer.timeout.connect(self.calculate_score)

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

                # 피드백 텍스트를 직접 그립니다.
                if self.feedback:
                    painter = QPainter(pixmap)
                    painter.setRenderHint(QPainter.Antialiasing)
                    painter.setRenderHint(QPainter.TextAntialiasing)

                    # 폰트 크기 비율을 5로 되돌립니다.
                    font_size = max(30, int(pixmap.height() / 5))
                    font = QFont("Arial", font_size, QFont.Bold)
                    painter.setFont(font)

                    if self.feedback == "GOOD":
                        painter.setPen(QColor(0, 255, 0, int(self.current_opacity * 255)))
                    else:
                        painter.setPen(QColor(255, 0, 0, int(self.current_opacity * 255)))

                    # 글씨 잘림 현상을 해결하기 위해 텍스트 영역의 높이를 더 크게 설정
                    text_rect = QRect(0, 0, pixmap.width(), int(pixmap.height() * 0.3))
                    painter.drawText(text_rect, Qt.AlignLeft | Qt.AlignTop, self.feedback)
                    painter.end()

                self.cam_label.setPixmap(
                    pixmap.scaled(self.cam_label.size(),
                                 Qt.KeepAspectRatio,
                                 Qt.SmoothTransformation)
                )

                # 카운트다운 라벨이 보일 때만 위치 재설정 (중앙)
                if not self.overlay_label.isHidden():
                    self.overlay_label.setGeometry(self.cam_label.rect())

            except Exception as e:
                print(f"Exception during frame processing: {e}")
                self.cam_label.setText(f"처리 중 오류 발생: {e}")
                self.cam_label.setStyleSheet("background-color: black; color: white; font-size: 20px;")
                if self.frame_timer:
                    self.frame_timer.stop()

    def calculate_score(self):
        if not self.cap or not self.cap.isOpened() or self.count > 0:
            return

        ret, frame = self.cap.read()
        if not ret or frame is None:
            return

        cam_kps, cam_conf = self.infer_pose(frame)

        new_feedback = ""
        if cam_kps is not None and len(self.reference_data) > 0:
            ref_frame_index = self.player.position() // 1000 * 30
            if ref_frame_index < len(self.reference_data):
                ref_data = self.reference_data[ref_frame_index]
                ref_kps = np.array(ref_data["kps"])

                cam_kps_norm = normalize_keypoints(cam_kps)
                ref_kps_norm = normalize_keypoints(ref_kps)

                vec_ref = pose_to_anglevec(ref_kps_norm)
                vec_live = pose_to_anglevec(cam_kps_norm)

                score, _, _ = frame_score_strict(vec_ref, vec_live)

                if score > 0.5:
                    new_feedback = "GOOD"
                else:
                    new_feedback = "BAD"
            else:
                 new_feedback = ""

        if new_feedback and new_feedback != self.feedback:
            self.feedback = new_feedback
            self.current_opacity = 1.0
            self.fade_timer.start(50)

    def update_fade(self):
        """투명도를 점진적으로 낮춰 텍스트를 사라지게 합니다."""
        self.current_opacity -= self.fade_step
        if self.current_opacity <= 0:
            self.current_opacity = 0
            self.fade_timer.stop()
            self.feedback = "" # 피드백 텍스트 자체를 지웁니다.
            self.update_frame() # 텍스트가 사라진 후 프레임을 다시 그립니다.

        # 텍스트가 페이드아웃 되는 동안 계속 업데이트
        self.update_frame()

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
            self.score_timer.start(200)

    def play_video(self):
        abs_path = QFileInfo(self.args.ref).absoluteFilePath()
        self.player.setMedia(QMediaContent(QUrl.fromLocalFile(abs_path)))
        self.player.play()

    def update_overlay_size_and_position(self):
        pass

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QEvent.WindowStateChange:
            QTimer.singleShot(0, self.equalize_splitter)

    def resizeEvent(self, event):
        QTimer.singleShot(0, self.equalize_splitter)
        super().resizeEvent(event)
        if self.count >= 0:
            self.overlay_label.setGeometry(self.cam_label.rect())

    def closeEvent(self, event):
        if self.frame_timer:
            self.frame_timer.stop()
        if self.score_timer:
            self.score_timer.stop()
        if self.cap and self.cap.isOpened():
            self.cap.release()
        super().closeEvent(event)
