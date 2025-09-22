#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import json
import numpy as np
import time
import os
from PyQt5.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QSizePolicy,
    QSplitter, QMessageBox, QGraphicsOpacityEffect, QHBoxLayout
)
from PyQt5.QtMultimedia import QMediaPlayer
from PyQt5.QtCore import QTimer, Qt, QRect, QPropertyAnimation, QEasingCurve, QCoreApplication
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPainter

# BasePoseApp 클래스를 임포트합니다.
from .base_pose_app import BasePoseApp
# 싱글 플레이어 모드에 필요한 추가 모듈을 임포트합니다.
from core.model_loader import make_infer
from core.pose_utils import (
    normalize_keypoints, pose_to_anglevec, frame_score_strict
)
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtCore import pyqtSignal

from core.person_utils import get_person_center, classify_region

class SinglePlayerApp(BasePoseApp):
    """
    BasePoseApp을 상속받아 싱글 플레이어 모드 로직을 구현한 클래스.
    """
    def __init__(self, args, model, use_half, ser):
        # 부모 클래스의 생성자를 호출하여 기본 UI를 설정합니다.
        super().__init__(args)
        
        self.model = model
        self.use_half = use_half
        
        self.ser = ser
        
        self.button_container = None
        self.game_over_flag = False
        self.cam_kps = None # 포즈 감지 결과를 저장할 변수

        # 영상 녹화 관련 변수
        self.video_writer = None
        resource_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'resource')
        if not os.path.exists(resource_dir):
            os.makedirs(resource_dir)
        self.output_path = os.path.join(resource_dir, 'output.mp4')

        # 포즈 감지 모델 로딩
        self.infer_pose = make_infer(self.model, self.args, self.use_half)
        print("Using pre-loaded single-person pose detection model.")

        # JSON 데이터 로딩
        if self.args.json is None:
            error_message = "Error: JSON file path not provided. Please provide a path using the --json argument."
            QMessageBox.critical(self, "Error", error_message)
            self.close()
            return
        
        try:
            with open(self.args.json, 'r') as f:
                self.reference_data = json.load(f)["frames"]
                print(f"Successfully loaded {len(self.reference_data)} reference frames.")
        except FileNotFoundError:
            error_message = f"Error: Reference JSON file '{self.args.json}' not found."
            QMessageBox.critical(self, "Error", error_message)
            self.close()
            return

        # 피드백 텍스트를 위한 QLabel 추가
        self.feedback_label = QLabel(self.cam_label)
        self.feedback_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.feedback_label.setContentsMargins(20, 20, 0, 0)
        self.feedback_label.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.feedback_label.hide()

        # 피드백 애니메이션
        self.feedback_opacity_effect = QGraphicsOpacityEffect(self.feedback_label)
        self.feedback_label.setGraphicsEffect(self.feedback_opacity_effect)
        self.fade_animation = QPropertyAnimation(self.feedback_opacity_effect, b"opacity")
        self.fade_animation.setDuration(1500)
        self.fade_animation.setStartValue(1.0)
        self.fade_animation.setEndValue(0.0)
        self.fade_animation.setEasingCurve(QEasingCurve.OutQuad)
        self.fade_animation.finished.connect(self.feedback_label.hide)

        # 게임 상태 변수 초기화
        self.count = 6
        self.final_score = 80
        self.score_history = []
        self.score_history_length = 3
        self.follow_delay_ms = 200

        self.count_timer.start(1000)
        self.score_timer.timeout.connect(self.calculate_score)

        self.init_webcam()
        if self.args.ref is not None:
            self.show_preview_frame(self.args.ref)

    def update_countdown(self):
        """
        카운트다운을 업데이트하고, 카운트다운이 끝나면 게임을 시작합니다.
        """
        self.count -= 1
        if self.count > 0:
            self.overlay_label.setText(str(self.count))
        elif self.count == 0:
            self.overlay_label.setText("START")
            # 녹화 시작
            if self.cap and self.cap.isOpened():
                width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = self.cap.get(cv2.CAP_PROP_FPS)
                if fps == 0:
                    fps = 30 # 기본 FPS
                self.video_writer = cv2.VideoWriter(self.output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
                print(f"🎥 웹캠 녹화를 시작합니다. 저장 경로: {self.output_path}")
        else:
            self.overlay_label.hide()
            self.count_timer.stop()
            self.video_stack.setCurrentWidget(self.video_widget)
            QTimer.singleShot(0, self.equalize_splitter)
            self.play_video()
            self.score_timer.start(333)

    def update_frame(self):
        """웹캠 프레임을 업데이트하고, 녹화하며, 포즈 감지를 수행합니다."""
        if self.game_over_flag:
            self.display_final_score()
            return

        if not self.cap or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret or frame is None:
            print("Error: 웹캠에서 프레임을 읽어올 수 없습니다.")
            return

        # 좌우반전 추가
        frame = cv2.flip(frame, 1)

        # 프레임 녹화
        if self.video_writer:
            self.video_writer.write(frame)

        # 게임 시작 후에만 포즈 감지 수행
        # if self.count <= 0:
        #     self.cam_kps, _ = self.infer_pose(frame)
        if self.count <= 0:
            self.cam_kps, _ = self.infer_pose(frame)
            if self.cam_kps is not None:
                # 중심 좌표 구하기
                center = get_person_center(self.cam_kps)
                if center is not None:
                    cx, cy = center
                    region = classify_region(cx, frame.shape[1])
                    # print(f"[Single] 중심 x={cx:.1f}, 화면폭={frame.shape[1]}, 영역={region}")
                    self.ser.write(region.encode())

        # 화면에 프레임 표시 (BasePoseApp 로직과 유사)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.cam_label.setPixmap(pixmap.scaled(self.cam_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        # 피드백 라벨 위치 및 폰트 업데이트
        self.feedback_label.setGeometry(10, 10, int(self.cam_label.width() / 1.5), int(self.cam_label.height() / 3))
        self.feedback_label.setFont(QFont("Arial", int(self.cam_label.height() / 15), QFont.Bold))

    def calculate_score(self):
        """update_frame에서 감지된 포즈를 사용하여 점수를 계산합니다."""
        if self.cam_kps is None or self.count > 0 or self.game_over_flag:
            return

        current_score = -1.0
        
        if len(self.reference_data) > 0:
            delayed_position = self.player.position() - self.follow_delay_ms
            if delayed_position < 0: delayed_position = 0
            ref_frame_index = int(delayed_position / 1000 * 30)
            ref_data_index = ref_frame_index // 10

            if ref_data_index < len(self.reference_data):
                ref_data = self.reference_data[ref_data_index]
                ref_kps = np.array(ref_data["kps"])

                cam_kps_norm = normalize_keypoints(self.cam_kps)
                ref_kps_norm = normalize_keypoints(ref_kps)

                vec_ref = pose_to_anglevec(ref_kps_norm)
                vec_live = pose_to_anglevec(cam_kps_norm)

                score, _, _ = frame_score_strict(vec_ref, vec_live)
                current_score = score
        
        if current_score != -1.0:
            self.score_history.append(current_score)

        if len(self.score_history) >= self.score_history_length:
            smoothed_score = np.mean(self.score_history)

            new_feedback = ""
            if smoothed_score >= 80.0: new_feedback = "PERFECT"
            elif smoothed_score >= 50.0: new_feedback = "GOOD"
            else: new_feedback = "BAD"
            
            if new_feedback == "PERFECT": self.final_score = min(100, self.final_score + 1)
            elif new_feedback == "BAD": self.final_score = max(0, self.final_score - 1)
            
            self.feedback = new_feedback
            self.feedback_label.setText(new_feedback)
            if new_feedback == "PERFECT": self.feedback_label.setStyleSheet("color: lime; font-weight: bold;")
            elif new_feedback == "GOOD": self.feedback_label.setStyleSheet("color: yellow; font-weight: bold;")
            else: self.feedback_label.setStyleSheet("color: red; font-weight: bold;")
            
            self.fade_animation.stop()
            self.feedback_label.show()
            self.feedback_opacity_effect.setOpacity(1.0)
            self.fade_animation.start()

            self.score_history = []

    def handle_video_state(self, state):
        """부모 클래스의 비디오 상태 감지 메서드를 오버라이드하여 게임 종료를 처리합니다."""
        if state == QMediaPlayer.StoppedState and self.player.duration() > 0:
            print(f"🏁 비디오 재생 종료. 최종 점수: {int(self.final_score)}")
            self.game_over_flag = True
            
            # 녹화 종료
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
                print(f"✅ 영상이 성공적으로 저장되었습니다: {self.output_path}")

            self.close()

    def closeEvent(self, event):
        """창이 닫힐 때 호출되는 이벤트 핸들러."""
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            print("ℹ️ 창이 닫혀 녹화를 중지하고 영상을 저장했습니다.")
        super().closeEvent(event)

    def display_final_score(self):
        """최종 점수를 화면에 그립니다."""
        label_size = self.cam_label.size()
        if label_size.width() <= 1 or label_size.height() <= 1:
            QTimer.singleShot(50, self.display_final_score)
            return

        pixmap = QPixmap(label_size)
        pixmap.fill(QColor(0, 0, 0))
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        title_font = QFont("Arial", 40, QFont.Bold)
        painter.setFont(title_font)
        painter.setPen(QColor(255, 255, 255))
        title_rect = QRect(0, int(pixmap.height() * 0.2), pixmap.width(), 50)
        painter.drawText(title_rect, Qt.AlignCenter, "FINAL SCORE")

        score_font = QFont("Arial", 120, QFont.Bold)
        painter.setFont(score_font)
        score_to_display = int(self.final_score)

        if score_to_display >= 80: painter.setPen(QColor(0, 255, 0))
        elif score_to_display >= 50: painter.setPen(QColor(255, 255, 0))
        else: painter.setPen(QColor(255, 0, 0))
        
        score_rect = QRect(0, int(pixmap.height() * 0.4), pixmap.width(), 200)
        painter.drawText(score_rect, Qt.AlignCenter, str(score_to_display))
        
        painter.end()
        self.cam_label.setPixmap(pixmap)

        if self.button_container is None:
            self.button_container = QWidget(self.cam_label)
            layout = QHBoxLayout(self.button_container)
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
                0, int(self.cam_label.height() * 0.8), self.cam_label.width(), 60
            )
            self.button_container.show()
            self.main_btn.clicked.connect(self.goMainRequested.emit)
            self.rank_btn.clicked.connect(self.goRankRequested.emit)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.count > 0:
            self.overlay_label.setGeometry(self.cam_label.rect())
            overlay_font_size = int(self.cam_label.height() / 5)
            self.overlay_label.setFont(QFont("Arial", overlay_font_size, QFont.Bold))
        if self.game_over_flag:
            self.display_final_score()
            if self.button_container:
                self.button_container.setGeometry(
                    0, int(self.cam_label.height() * 0.8),
                    self.cam_label.width(), 60
                )
