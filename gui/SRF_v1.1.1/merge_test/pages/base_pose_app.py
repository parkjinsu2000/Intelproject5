#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2
from PyQt5.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QSizePolicy,
    QStackedWidget, QSplitter, QMessageBox, QHBoxLayout
)
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import QTimer, Qt, QUrl, QFileInfo, QSize, QEvent, pyqtSignal, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QImage, QPixmap, QFont

class MyVideoWidget(QVideoWidget):
    """QVideoWidget을 상속받아 sizeHint를 오버라이드하여 레이아웃 내에서 유연하게 크기 조절"""
    def sizeHint(self):
        return QSize(1, 1)

class BasePoseApp(QWidget):
    """
    포즈 교정 앱의 베이스 UI 클래스:
    왼쪽(참고 영상)과 오른쪽(웹캠) 화면을 QSplitter로 분할하여 관리
    """
    goMainRequested = pyqtSignal()
    goRankRequested = pyqtSignal()

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.setMinimumSize(400, 300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.cap_index = 0
        self.cap = None
        self.game_started = False
        self.game_over = False
        self.count = 6 # 카운트다운 시작 값

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

        # 카운트다운 오버레이
        self.overlay_label = QLabel("", self.cam_label)
        self.overlay_label.setAlignment(Qt.AlignCenter)
        self.overlay_label.setStyleSheet("color: red; font-weight: bold; background-color: transparent;")
        self.overlay_label.setFont(QFont("Arial", 96))
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
        root_layout.addWidget(self.splitter)
        
        # --- 4. 타이머 및 상태 변수 ---
        self.frame_timer = None
        self.count_timer = QTimer(self)
        self.score_timer = QTimer(self)
        self.count_timer.timeout.connect(self.update_countdown)

        # 웹캠, 비디오, 스플리터 초기화는 상속 클래스에서 호출
        QTimer.singleShot(0, self.equalize_splitter)

    def init_webcam(self):
        """웹캠을 초기화하고 프레임 읽기 타이머를 시작합니다."""
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
        """스플리터의 화면 비율을 균등하게 조정합니다."""
        w = max(2, self.splitter.width())
        self.splitter.setSizes([w // 2, w - (w // 2)])
    
    def show_preview_frame(self, video_path):
        """비디오의 첫 프레임을 미리보기 화면에 표시합니다."""
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

    def play_video(self):
        """참고 비디오를 재생합니다."""
        if self.args.ref:
            abs_path = QFileInfo(self.args.ref).absoluteFilePath()
            self.player.setMedia(QMediaContent(QUrl.fromLocalFile(abs_path)))
            self.player.play()

    def handle_video_state(self, state):
        """비디오 재생 상태 변화를 감지하여 게임 종료를 처리합니다."""
        if state == QMediaPlayer.StoppedState:
            self.game_over()

    def update_frame(self):
        """웹캠에서 프레임을 읽어와 화면에 표시합니다. 상속 클래스에서 포즈 감지 및 점수 계산 로직을 추가합니다."""
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret or frame is None:
                print("Error: 웹캠에서 프레임을 읽어올 수 없습니다. 재연결을 시도합니다.")
                self.cam_label.setText("웹캠에서 프레임을 읽어올 수 없습니다. 재연결 중...")
                self.init_webcam()
                return

            # 웹캠 프레임의 좌우 반전을 제거했습니다.

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape

            qt_image = QImage(frame_rgb.data, w, h, ch * w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)

            self.cam_label.setPixmap(
                pixmap.scaled(self.cam_label.size(),
                             Qt.KeepAspectRatio,
                             Qt.SmoothTransformation)
            )

            # 오버레이 라벨 위치를 웹캠 라벨 크기에 맞게 조정
            if not self.overlay_label.isHidden():
                self.overlay_label.setGeometry(self.cam_label.rect())
                overlay_font_size = int(self.cam_label.height() / 5)
                self.overlay_label.setFont(QFont("Arial", overlay_font_size, QFont.Bold))
    
    def game_over(self):
        """게임 종료 시 호출될 메서드. 상속 클래스에서 오버라이드합니다."""
        if self.frame_timer:
            self.frame_timer.stop()
        if self.score_timer:
            self.score_timer.stop()
        if self.cap and self.cap.isOpened():
            self.cap.release()
        print("게임이 종료되었습니다. 최종 점수를 표시합니다.")

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QEvent.WindowStateChange:
            QTimer.singleShot(0, self.equalize_splitter)

    def resizeEvent(self, event):
        QTimer.singleShot(0, self.equalize_splitter)
        super().resizeEvent(event)
        # resize 이벤트 발생 시 오버레이 라벨 위치를 재조정
        if not self.overlay_label.isHidden():
            self.overlay_label.setGeometry(self.cam_label.rect())
            overlay_font_size = int(self.cam_label.height() / 5)
            self.overlay_label.setFont(QFont("Arial", overlay_font_size, QFont.Bold))

    def update_countdown(self):
        """카운트다운을 처리하고 게임을 시작합니다."""
        self.count -= 1
        self.overlay_label.show()
        if self.count > 0:
            self.overlay_label.setText(str(self.count))
        elif self.count == 0:
            self.overlay_label.setText("START!")
            self.video_stack.setCurrentWidget(self.video_widget)
            self.play_video()
        else:
            self.overlay_label.hide()
            self.game_started = True
            self.count_timer.stop()
