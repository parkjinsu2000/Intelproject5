#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import json
import numpy as np
import time
from PyQt5.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QSizePolicy,
    QSplitter, QMessageBox, QHBoxLayout
)
from PyQt5.QtMultimedia import QMediaPlayer
from PyQt5.QtCore import QTimer, Qt, QRect, QCoreApplication, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPainter
from PyQt5.QtWidgets import QPushButton
import random
import threading
import sys
import torch
from ultralytics import YOLO
import collections

# BasePoseApp 클래스를 임포트합니다.
from .base_pose_app import BasePoseApp
# 포즈 감지 및 유틸리티 모듈을 임포트합니다.
from core.pose_utils import (
    normalize_keypoints, pose_to_anglevec, frame_score_strict, draw_pose
)

# YOLO 모델 설정
MODEL_PATH_DEFAULT = "yolov8m-pose.pt"
DETECT_CONF_THRES = 0.25
KPT_CONF_THRES = 0.20

# 키포인트 쌍 (왼쪽 <-> 오른쪽)
FLIP_MAP = [
    [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]
]

class MultiPlayerApp(BasePoseApp):
    """
    BasePoseApp을 상속받아 멀티 플레이어 모드 로직을 구현한 클래스.
    Firebase를 사용하지 않고 로컬에서만 작동하도록 수정되었습니다.
    """
    goMainRequested = pyqtSignal()
    goRankRequested = pyqtSignal()
    updateDisplaySignal = pyqtSignal()

    def __init__(self, args, model, use_half, player_count=2):
        super().__init__(args)
        
        self.game_over_flag = False
        self.model = model
        self.use_half = use_half
        self.button_container = None
        self.tracker_yaml = "botsort.yaml"
        self.active_players = {} # 플레이어 ID (1 또는 2) -> 정보
        self.previous_kps = {} # 각 플레이어의 이전 키포인트 저장

        if self.args.json is None:
            QMessageBox.critical(self, "오류", "오류: JSON 파일 경로가 제공되지 않았습니다.")
            self.close()
            return
        
        try:
            with open(self.args.json, 'r') as f:
                self.reference_data = json.load(f)["frames"]
                print(f"{len(self.reference_data)}개의 참조 프레임을 성공적으로 로드했습니다.")
        except FileNotFoundError:
            QMessageBox.critical(self, "오류", f"오류: 참조 JSON 파일 '{self.args.json}'을 찾을 수 없습니다.")
            self.close()
            return
        
        # 실시간 피드백 관련 UI 요소 삭제
        self.player_info_label = QLabel(self)
        self.player_info_label.setAlignment(Qt.AlignTop | Qt.AlignRight)
        self.player_info_label.setStyleSheet("color: white; background-color: rgba(0,0,0,150); padding: 10px;")
        self.player_info_label.hide()

        self.count = 6
        self.final_score = 80.0
        self.score_history = collections.defaultdict(list)
        self.score_history_length = 3
        self.follow_delay_ms = 200
        self.start_time = None
        self.end_time = None
        
        self.local_scores = collections.defaultdict(lambda: 80)
        self.player_rank_map = {}

        self.count_timer.start(1000)
        self.score_timer.timeout.connect(self.calculate_score)
        
        self.init_webcam()
        if self.args.ref is not None:
            self.show_preview_frame(self.args.ref)
    
    def play_video(self):
        super().play_video()
        self.start_time = time.time()

    def _flip_keypoints(self, kps):
        """키포인트 쌍을 좌우 반전시킵니다."""
        kps = kps.copy()
        for i, j in FLIP_MAP:
            kps[i], kps[j] = kps[j].copy(), kps[i].copy()
        return kps

    def update_player_info_display(self):
        """실제 플레이어의 점수를 화면에 표시합니다. 시각적으로는 표시하지 않습니다."""
        info_text = ""
        # 점수를 기준으로 정렬하고 순위 부여
        sorted_players = sorted(self.local_scores.items(), key=lambda item: item[1], reverse=True)
        
        # ID와 순위 매핑 업데이트
        player_ranks = {}
        for i, (player_id, score) in enumerate(sorted_players):
            if i > 0 and score == sorted_players[i-1][1]:
                player_ranks[player_id] = player_ranks[sorted_players[i-1][0]]
            else:
                player_ranks[player_id] = i + 1
        self.player_rank_map = player_ranks

        for player_id, score in sorted_players:
            score_to_display = int(score)
            display_name = f"Player {player_id}"
            info_text += f"{display_name}: {score_to_display}\n"
        
        # 시각적 표시만 제거
        self.player_info_label.setText(info_text)
        self.player_info_label.hide() # 항상 숨김

    def infer_and_track_once(self, model, frame, tracker_yaml, imgsz, device, half):
        """
        멀티 플레이어 추적을 위한 함수입니다.
        반환: dict {track_id: (kps_xy(17,2), kps_conf(17,), box_xyxy(4,))}
        """
        with torch.inference_mode():
            results = model.track(frame, imgsz=imgsz, device=device, half=half,
                                  conf=DETECT_CONF_THRES, verbose=False,
                                  persist=True, tracker=tracker_yaml, stream=False)
        if not results:
            return {}
        res = results[0]
        if (res.keypoints is None) or (len(res.keypoints)==0):
            return {}
        
        ids = getattr(res.boxes, "id", None)
        out = {}
        if ids is not None:
            ids = ids.detach().cpu().numpy().astype(int)
            boxes = res.boxes.xyxy.detach().cpu().numpy()
            for i in range(len(ids)):
                tid = int(ids[i])
                kps = res.keypoints.xy[i].detach().cpu().numpy()
                conf = res.keypoints.conf[i].detach().cpu().numpy()
                kps[conf < KPT_CONF_THRES] = np.nan
                box = boxes[i].astype(float)
                out[tid] = (kps, conf, box)
        else:
            boxes = res.boxes.xyxy.detach().cpu().numpy()
            for i in range(len(boxes)):
                tid = i
                kps = res.keypoints.xy[i].detach().cpu().numpy()
                conf = res.keypoints.conf[i].detach().cpu().numpy()
                kps[conf < KPT_CONF_THRES] = np.nan
                box = boxes[i].astype(float)
                out[tid] = (kps, conf, box)
                
        return out

    def update_frame(self, force_refresh=False):
        """웹캠 프레임을 업데이트하고 포즈 감지 결과를 화면에 표시합니다."""
        if self.game_over_flag:
            self.display_final_score()
            return

        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret or frame is None:
                return
            
            flipped_frame = cv2.flip(frame, 1)
            tracked_players = self.infer_and_track_once(self.model, flipped_frame, self.tracker_yaml, self.args.imgsz, self.args.device, self.use_half)
            display_frame = flipped_frame.copy()

            # 플레이어를 x축 기준으로 정렬하여 Player 1과 Player 2를 결정합니다.
            all_detected = list(tracked_players.values())
            # box의 x1 좌표(p[2][0])를 기준으로 정렬합니다.
            all_detected.sort(key=lambda p: p[2][0])
            
            new_active_players = {}
            # 가장 왼쪽에 있는 플레이어를 Player 1로 할당합니다.
            if len(all_detected) > 0:
                kps, _, box = all_detected[0]
                new_active_players[1] = {'tid': 1, 'kps': kps, 'box': box}
            
            # 두 번째로 왼쪽에 있는 플레이어를 Player 2로 할당합니다.
            if len(all_detected) > 1:
                kps, _, box = all_detected[1]
                new_active_players[2] = {'tid': 2, 'kps': kps, 'box': box}
            
            self.active_players = new_active_players

            self.draw_all_poses(display_frame)

            frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            qt_image = QImage(frame_rgb.data, w, h, ch * w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            self.cam_label.setPixmap(
                pixmap.scaled(self.cam_label.size(),
                             Qt.KeepAspectRatio,
                             Qt.SmoothTransformation)
            )

    def draw_all_poses(self, frame):
        """
        추적된 모든 플레이어의 포즈와 ID를 그립니다.
        """
        for player_id, player_data in self.active_players.items():
            kps = player_data['kps']
            box = player_data['box']
            if kps is not None and box is not None:
                display_name = f"Player {player_id}"
                
                # 바운딩 박스를 그립니다.
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                # 포즈 선을 그립니다.
                draw_pose(frame, kps)
                
                # 텍스트를 그립니다.
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                thickness = 2
                text_size = cv2.getTextSize(display_name, font, font_scale, thickness)[0]
                text_x = int(x1)
                text_y = int(y1) - 10
                
                # 텍스트 배경을 그립니다.
                cv2.rectangle(frame, (text_x, text_y - text_size[1] - 5), (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)
                cv2.putText(frame, display_name, (text_x + 2, text_y - 2), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


    def calculate_score(self):
        """웹캠 프레임과 참고 포즈를 비교하여 모든 플레이어의 점수를 계산합니다."""
        if not self.cap or not self.cap.isOpened() or self.count > 0 or self.game_over_flag:
            return

        ret, frame = self.cap.read()
        if not ret or frame is None:
            return

        # 카메라 프레임을 좌우 반전합니다. 이 프레임으로 포즈를 감지하고 점수를 계산합니다.
        flipped_frame = cv2.flip(frame, 1)
      
        

        tracked_players = self.infer_and_track_once(self.model, flipped_frame, self.tracker_yaml, self.args.imgsz, self.args.device, self.use_half)
        
        # 플레이어를 x축 기준으로 정렬하여 Player 1과 Player 2를 결정하고 점수를 계산합니다.
        all_detected = list(tracked_players.values())
        all_detected.sort(key=lambda p: p[2][0]) # box의 x1 좌표(p[2][0])를 기준으로 정렬

        # 최대 2명의 플레이어에 대해 점수를 계산합니다.
        for i, (cam_kps, cam_conf, _) in enumerate(all_detected[:2]):
            player_id = i + 1 # Player 1 또는 Player 2

            if cam_kps is not None and cam_kps.size > 0 and len(self.reference_data) > 0:
                delayed_position = self.player.position() - self.follow_delay_ms
                if delayed_position < 0: delayed_position = 0
                ref_frame_index = int(delayed_position / 1000 * 30)
                ref_data_index = ref_frame_index // 10

                if ref_data_index < len(self.reference_data):
                    ref_data = self.reference_data[ref_data_index]
                    ref_kps = np.array(ref_data["kps"])

                    cam_kps_norm = normalize_keypoints(cam_kps)
                    ref_kps_norm = normalize_keypoints(ref_kps)

                    vec_ref = pose_to_anglevec(ref_kps_norm)
                    vec_live = pose_to_anglevec(cam_kps_norm)
                    
                    # --- 점수 가중치 적용 로직 시작 ---
                    num_angles = len(vec_ref)
                    weights = np.ones(num_angles)
                    weights[0:4] = 2.0
                    vec_ref_weighted = vec_ref * weights
                    vec_live_weighted = vec_live * weights
                    current_score, _, _ = frame_score_strict(vec_ref_weighted, vec_live_weighted)
                    # --- 점수 가중치 적용 로직 끝 ---
                    
                    # --- 정지 페널티 로직 추가 시작 ---
                    if player_id in self.previous_kps and self.previous_kps[player_id] is not None:
                        kps_diff = np.linalg.norm(cam_kps - self.previous_kps[player_id])
                        movement_threshold = 20.0
                        if kps_diff < movement_threshold:
                            current_score -= 5
                            if current_score < 0:
                                current_score = 0
                    
                    self.previous_kps[player_id] = cam_kps.copy()
                    # --- 정지 페널티 로직 추가 끝 ---
                    
                    if current_score != -1.0:
                        self.score_history[player_id].append(current_score)
                    
                    if len(self.score_history[player_id]) >= self.score_history_length:
                        smoothed_score = np.mean(self.score_history[player_id])
                        
                        current_total_score = self.local_scores[player_id]
                        if smoothed_score >= 70.0:
                            self.local_scores[player_id] = min(100, current_total_score + 1)
                        elif smoothed_score < 30.0:
                            self.local_scores[player_id] = max(0, current_total_score - 1)
                        
                        self.score_history[player_id] = []
        
        self.update_player_info_display()

    def update_countdown(self):
        """
        카운트다운을 업데이트하고, 카운트다운이 끝나면 게임을 시작합니다.
        """
        self.count -= 1
        overlay_font_size = int(self.cam_label.height() / 5)
        self.overlay_label.setFont(QFont("Arial", overlay_font_size, QFont.Bold))
        self.overlay_label.setGeometry(self.cam_label.rect())
        
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
            self.score_timer.start(333)

    def handle_video_state(self, state):
        """부모 클래스의 비디오 상태 감지 메서드를 오버라이드하여 게임 종료를 처리합니다."""
        if state == QMediaPlayer.StoppedState:
            print("비디오 재생이 종료되었습니다. 최종 점수를 표시합니다.")
            self.game_over_flag = True
            super().game_over()
            self.display_final_score()
            self.end_time = time.time() # 게임 종료 시간 기록

    def display_final_score(self):
        """
        최종 점수를 화면에 그립니다.
        각 플레이어의 최종 점수를 나열하여 표시합니다.
        """
        label_size = self.cam_label.size()
        if label_size.width() <= 1 or label_size.height() <= 1:
            QTimer.singleShot(50, self.display_final_score)
            return

        pixmap = QPixmap(label_size)
        pixmap.fill(QColor(0, 0, 0))
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.TextAntialiasing)
        
        title_font = QFont("Arial", 40, QFont.Bold)
        painter.setFont(title_font)
        painter.setPen(QColor(255, 255, 255))
        title_rect = QRect(0, int(pixmap.height() * 0.2), pixmap.width(), 50)
        painter.drawText(title_rect, Qt.AlignCenter, "FINAL SCORES")

        sorted_scores = sorted(self.local_scores.items(), key=lambda item: item[1], reverse=True)
        player_rank_map = {}
        for i, (player_id, score) in enumerate(sorted_scores):
            if i > 0 and score == sorted_scores[i-1][1]:
                player_rank_map[player_id] = player_rank_map[sorted_scores[i-1][0]]
            else:
                player_rank_map[player_id] = i + 1
        
        score_font = QFont("Arial", 30, QFont.Bold)
        painter.setFont(score_font)
        line_height = 40
        start_y = int(pixmap.height() * 0.4)

        for i, (player_id, score) in enumerate(sorted_scores):
            score_to_display = int(score)
            display_name = f"Player {player_id}"

            if score_to_display >= 80:
                painter.setPen(QColor(0, 255, 0))
            elif score_to_display >= 50:
                painter.setPen(QColor(255, 255, 0))
            else:
                painter.setPen(QColor(255, 0, 0))

            text = f"{display_name}: {score_to_display}"
            text_y = start_y + i * line_height
            score_rect = QRect(0, text_y, pixmap.width(), line_height)
            painter.drawText(score_rect, Qt.AlignCenter, text)

        painter.end()
        self.cam_label.setPixmap(pixmap)

        if self.button_container is None:
            self.button_container = QWidget(self.cam_label)
            layout = QHBoxLayout(self.button_container)
            self.main_btn = QPushButton("메인으로")
            font = QFont("Arial", 20, QFont.Bold)
            self.main_btn.setFont(font)
            self.main_btn.setStyleSheet("background-color: white; padding: 10px; border-radius: 10px;")
            layout.addStretch(1)
            layout.addWidget(self.main_btn)
            layout.addStretch(1)
            self.button_container.setLayout(layout)
            self.button_container.setGeometry(
                0, int(self.cam_label.height() * 0.8),
                self.cam_label.width(), 60
            )
            self.button_container.show()
            self.main_btn.clicked.connect(self.goMainRequested.emit)

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
