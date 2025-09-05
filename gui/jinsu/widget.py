# widget.py

import cv2
import numpy as np
import json
import os
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QGraphicsView, QGraphicsScene,
    QPushButton, QMessageBox, QHBoxLayout
)
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent, QAudioOutput
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import QUrl, QFileInfo
from PyQt5.QtGui import QPixmap, QImage, QPainter
from PyQt5.QtCore import QTimer, Qt
from pose_utils import (
    normalize_keypoints,
    flip_horizontal_pts,
    pose_to_anglevec,
    frame_score_strict,
    draw_pose
)
from model_loader import make_infer

class PoseScoreApp(QWidget):  # ✅ QWidget으로 변경
    def __init__(self, args, model, use_half):
        super().__init__()
        self.args = args
        self.model = model
        self.use_half = use_half
        self.infer_pose = make_infer(model, args, use_half)

        # 분석 상태 초기화
        self.total_points = 100
        self.ema_score = None
        self.ema_alpha = 0.25
        self.frame_idx = 0
        self.lastR = {"kps": None, "conf": None}
        self.lastU = {"kps": None, "conf": None}
        self.mirror = not args.no_mirror
        self.status_text = ""
        self.status_color = (255, 255, 255)
        self.writer = None

        # 카운트다운 설정
        self.countdown_texts = ["3", "2", "1", "START"]
        self.countdown_index = 0
        self.countdown_timer = QTimer()
        self.countdown_timer.timeout.connect(self.show_countdown_step)

        # UI 및 영상 초기화
        self.init_ui()
        self.init_video()
        self.start_countdown()

    def init_ui(self):

        self.resize(1280, 720)

        # 왼쪽: 정답 영상
        self.video_widget_left = QVideoWidget()
        self.player_left = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.player_left.setVideoOutput(self.video_widget_left)
        self.player_left.setVolume(100)
        self.video_widget_left.show()

        # 오른쪽: 실시간 분석 영상
        self.scene_right = QGraphicsScene(self)
        self.graphicsView_right = QGraphicsView(self.scene_right, self)
        self.graphicsView_right.setRenderHint(QPainter.Antialiasing)
        self.graphicsView_right.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView_right.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView_right.setFrameShape(QGraphicsView.NoFrame)

        layout = QHBoxLayout()
        layout.addWidget(self.video_widget_left)
        layout.addWidget(self.graphicsView_right)

        self.setLayout(layout)  # ✅ QWidget에서는 setCentralWidget 대신 setLayout 사용

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    # def init_video(self):
    #     self.capU = cv2.VideoCapture(self.args.cam)
    #     self.capR = cv2.VideoCapture(self.args.ref)
    #     if not self.capU.isOpened():
    #         QMessageBox.critical(self, "오류", "카메라 열기 실패")
    #         return

    #     if not self.capR.isOpened():
    #         QMessageBox.critical(self, "오류", "기준 영상 열기 실패")
    #         return

    #     W = int(max(self.capR.get(cv2.CAP_PROP_FRAME_WIDTH), 640))
    #     H = int(max(self.capR.get(cv2.CAP_PROP_FRAME_HEIGHT), 360))
    #     self.size_single = (W, H)

    #     abs_path = QFileInfo(self.args.ref).absoluteFilePath()
    #     self.player_left.setMedia(QMediaContent(QUrl.fromLocalFile(abs_path)))
    #     QTimer.singleShot(0, self.player_left.play)  # ✅ 렌더링 이후 재생

    def init_video(self):
        self.capU = cv2.VideoCapture(self.args.cam)
        self.capR = cv2.VideoCapture(self.args.ref)
        if not self.capU.isOpened():
            QMessageBox.critical(self, "오류", "카메라 열기 실패")
            return

        if not self.capR.isOpened():
            QMessageBox.critical(self, "오류", "기준 영상 열기 실패")
            return

        W = int(max(self.capR.get(cv2.CAP_PROP_FRAME_WIDTH), 640))
        H = int(max(self.capR.get(cv2.CAP_PROP_FRAME_HEIGHT), 360))
        self.size_single = (W, H)

        abs_path = QFileInfo(self.args.ref).absoluteFilePath()
        self.player_left.setMedia(QMediaContent(QUrl.fromLocalFile(abs_path)))
        QTimer.singleShot(0, self.player_left.play)  # ✅ 렌더링 이후 재생

        # --------- 여기 추가 ---------
        self.precomputed_ref_vecs = []
        cap_tmp = cv2.VideoCapture(self.args.ref)
        frame_count = int(cap_tmp.get(cv2.CAP_PROP_FRAME_COUNT))

        for i in range(frame_count):
            ok, frame = cap_tmp.read()
            if not ok:
                break
            frame = cv2.resize(frame, self.size_single)

            try:
                kpsR, confR = self.infer_pose(frame)
                if kpsR is None:
                    self.precomputed_ref_vecs.append(None)
                    continue
                kps_n = normalize_keypoints(kpsR)
                vec = pose_to_anglevec(kps_n)
                self.precomputed_ref_vecs.append(vec)
            except Exception as e:
                print(f"[warn] ref frame {i} 추론 실패:", e)
                self.precomputed_ref_vecs.append(None)

        cap_tmp.release()
        print(f"[info] precomputed_ref_vecs 준비 완료: {len(self.precomputed_ref_vecs)} 프레임")
        self.player_left.play()
        self.player_left.pause()

    def load_ref_vectors(self, json_path):
        try:
            with open(json_path, "r") as f:
                ref_data = json.load(f)
            frames = ref_data.get("frames", [])
            self.precomputed_ref_vecs = []
            for i, frame in enumerate(frames):
                if "kps" not in frame:
                    continue
                kps = np.array(frame["kps"])
                kps_n = normalize_keypoints(kps)
                vec = pose_to_anglevec(kps_n)
                self.precomputed_ref_vecs.append(vec)
        except Exception as e:
            QMessageBox.critical(self, "JSON 오류", f"정답 포즈 벡터 로딩 실패:\n{e}")
            self.precomputed_ref_vecs = []

    def start_countdown(self):
        self.frame_idx = 0
        self.ema_score = None
        self.total_points = 100
        self.lastR = {"kps": None, "conf": None}
        self.lastU = {"kps": None, "conf": None}
        self.status_text = ""
        self.status_color = (255, 255, 255)

        self.capR.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.capU.set(cv2.CAP_PROP_POS_FRAMES, 0)

        self.countdown_index = 0
        self.countdown_timer.start(1000)
        self.timer.start(30)

    def start_video(self):
        self.timer.start(30)  # 30ms마다 update_frame 호출

    def show_countdown_step(self):
        if self.countdown_index >= len(self.countdown_texts):
            # 카운트다운 종료: START 텍스트 유지 후 영상 재생
            self.countdown_timer.stop()
            self.display_start_frame()
            QTimer.singleShot(500, self.resume_video)  # 0.5초 후 재생 시작
            return

        txt = self.countdown_texts[self.countdown_index]
        self.countdown_index += 1

        okU, fU = self.capU.read()
        if not okU:
            return

        canvas = cv2.resize(fU, self.size_single)
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

        self.overlay_text(canvas, txt)
        self.update_canvas_right(canvas)

    def display_start_frame(self):
        okU, fU = self.capU.read()
        if not okU:
            return
        canvas = cv2.resize(fU, self.size_single)
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        self.overlay_text(canvas, "START")
        self.update_canvas_right(canvas)


    def overlay_text(self, canvas, txt):
        h, w = canvas.shape[:2]
        font_scale = h / 300.0
        thickness = int(font_scale * 2)
        size = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        org = (w // 2 - size[0] // 2, h // 2 + size[1] // 2)

        cv2.putText(canvas, txt, org, cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(canvas, txt, org, cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


    def resume_video(self):
        self.player_left.play()       # ✅ 영상 재생
        self.start_video()

    def update_frame(self):
        okR, fR = self.capR.read()
        okU, fU = self.capU.read()
        if not okR or not okU:
            self.timer.stop()
            return

        self.frame_idx += 1
        do_score = (self.frame_idx % self.args.every == 0)

        # 포즈 추론 및 점수 계산
        if do_score:
            try:
                kpsU, confU = self.infer_pose(fU)
                self.lastU = {"kps": kpsU, "conf": confU}
            except Exception as e:
                print(f"[warn] 추론 실패:", e)
                return

            if self.lastU["kps"] is not None:
                KU_use = flip_horizontal_pts(self.lastU["kps"]) if self.mirror else self.lastU["kps"]
                KU_n = normalize_keypoints(KU_use)
                vU = pose_to_anglevec(KU_n)

                if self.frame_idx >= len(self.precomputed_ref_vecs):
                    self.timer.stop()
                    return

                vR = self.precomputed_ref_vecs[self.frame_idx]
                s, _, _ = frame_score_strict(vR, vU)
                self.ema_score = s if self.ema_score is None else (
                    self.ema_score * (1 - self.ema_alpha) + s * self.ema_alpha)

                # 상태 텍스트 결정
                if s < 50.0:
                    self.total_points = max(0, self.total_points - 1)
                    self.status_text = "Bad"
                    self.status_color = (0, 0, 255)
                elif s >= 70.0:
                    self.total_points = min(100, self.total_points + 1)
                    self.status_text = "Good"
                    self.status_color = (0, 255, 0)
                else:
                    self.status_text = "Neutral"
                    self.status_color = (128, 128, 128)

        # 포즈 시각화
        if self.lastU["kps"] is not None and self.lastU["kps"].size > 0:
            draw_pose(fU, self.lastU["kps"], kps_conf=self.lastU["conf"])

        # 영상 준비
        canvas_right = cv2.resize(fU, self.size_single)
        canvas_right = cv2.cvtColor(canvas_right, cv2.COLOR_BGR2RGB)

        # 텍스트 오버레이
        if self.timer.isActive() and self.ema_score is not None:
            ch, cw = canvas_right.shape[:2]
            scale = ch / 400.0
            thick = int(scale * 2)

            # 점수 텍스트
            score_txt = f"Score: {self.ema_score:.1f}"
            sz = cv2.getTextSize(score_txt, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)[0]
            org = (cw // 2 - sz[0] // 2, int(ch * 0.08))
            cv2.putText(canvas_right, score_txt, org, cv2.FONT_HERSHEY_SIMPLEX,
                        scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
            cv2.putText(canvas_right, score_txt, org, cv2.FONT_HERSHEY_SIMPLEX,
                        scale, (255, 255, 255), thick, cv2.LINE_AA)

            # 상태 텍스트
            if self.status_text:
                sz = cv2.getTextSize(self.status_text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)[0]
                org = (cw // 2 - sz[0] // 2, int(ch * 0.90))
                cv2.putText(canvas_right, self.status_text, org, cv2.FONT_HERSHEY_SIMPLEX,
                            scale, (0, 0, 0), thick + 4, cv2.LINE_AA)
                cv2.putText(canvas_right, self.status_text, org, cv2.FONT_HERSHEY_SIMPLEX,
                            scale, self.status_color, thick, cv2.LINE_AA)

        # 화면에 출력
        self.update_canvas_right(canvas_right)




    def update_canvas_right(self, canvas):
        h, w = canvas.shape[:2]
        qimg = QImage(canvas.data, w, h, 3*w, QImage.Format_RGB888).rgbSwapped()
        pix = QPixmap.fromImage(qimg)
        self.scene_right.clear()
        self.scene_right.setSceneRect(0, 0, w, h)
        self.scene_right.addPixmap(pix)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)

    def closeEvent(self, ev):
        if self.capR: self.capR.release()
        if self.capU: self.capU.release()
        if self.writer: self.writer.release()
        print(f"[final] total score = {self.total_points}")
        ev.accept()
