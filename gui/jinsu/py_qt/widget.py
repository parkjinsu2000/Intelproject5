# widget.py

import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QGraphicsView, QGraphicsScene,
    QPushButton, QMessageBox, QVBoxLayout
)
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent, QAudioOutput
from PyQt5.QtCore import QUrl
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

class PoseScoreApp(QMainWindow):
    def __init__(self, args, model, use_half):
        super().__init__()
        self.last_canvas = None
        self.args = args
        self.model = model
        self.use_half = use_half
        self.infer_pose = make_infer(model, args, use_half)

        self.total_points = 100
        self.ema_score = None
        self.ema_alpha = 0.25
        self.frame_idx = 0

        self.audio_player = QMediaPlayer()
        self.audio_player.setMedia(QMediaContent(QUrl.fromLocalFile(self.args.ref)))
        self.audio_player.setVolume(100)
        self.audio_player.play()

        # 마지막으로 검출된 키포인트 저장
        self.lastR = {"kps": None, "conf": None}
        self.lastU = {"kps": None, "conf": None}
        self.mirror = not args.no_mirror

        # 상태 텍스트 (Good/Bad/Neutral)
        self.status_text = ""
        self.status_color = (255, 255, 255)

        # 카운트다운 설정
        self.countdown_texts = ["3", "2", "1", "START"]
        self.countdown_index = 0
        self.countdown_timer = QTimer()
        self.countdown_timer.timeout.connect(self.show_countdown_step)

        self.init_ui()
        self.init_video()
        self.start_countdown()


    def init_ui(self):
        self.setWindowTitle("Pose Score Viewer")
        self.resize(1280, 720)

        # QGraphicsView 세팅
        self.scene = QGraphicsScene(self)
        self.graphicsView = QGraphicsView(self.scene, self)
        self.graphicsView.setRenderHint(QPainter.Antialiasing)
        self.graphicsView.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView.setFrameShape(QGraphicsView.NoFrame)
        self.graphicsView.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.graphicsView.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)


        # 레이아웃
        layout = QVBoxLayout()
        layout.addWidget(self.graphicsView)
        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

        # 메인 타이머 (영상 프레임 업데이트)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def init_video(self):
        # 참조 영상과 카메라 열기
        self.capR = cv2.VideoCapture(self.args.ref)
        self.capU = cv2.VideoCapture(self.args.cam)
        if not self.capR.isOpened():
            QMessageBox.critical(self, "오류", f"기준 영상 열기 실패: {self.args.ref}")
            return
        if not self.capU.isOpened():
            QMessageBox.critical(self, "오류", "카메라 열기 실패")
            return

        # 최소 크기 강제
        W = int(max(self.capR.get(cv2.CAP_PROP_FRAME_WIDTH), 640))
        H = int(max(self.capR.get(cv2.CAP_PROP_FRAME_HEIGHT), 360))
        self.size_single = (W, H)

        # 저장 모드
        self.writer = None
        if self.args.save:
            fps = self.capR.get(cv2.CAP_PROP_FPS) or 30.0
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(self.args.save, fourcc, int(round(fps)), (W*2, H))

    def start_countdown(self):
        # 재시작을 위한 초기화
        self.frame_idx = 0
        self.ema_score = None
        self.total_points = 100
        self.lastR = {"kps": None, "conf": None}
        self.lastU = {"kps": None, "conf": None}
        self.status_text = ""
        self.status_color = (255, 255, 255)

        # 동영상 rewind
        self.capR.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.capU.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # 타이머들 초기화
        self.timer.stop()
        self.countdown_index = 0
        self.countdown_timer.start(1000)

    def show_countdown_step(self):
        if self.countdown_index >= len(self.countdown_texts):
            self.countdown_timer.stop()
            self.start_video()
            return

        txt = self.countdown_texts[self.countdown_index]
        self.countdown_index += 1

        okR, fR = self.capR.read()
        okU, fU = self.capU.read()
        if not okR or not okU:
            return

        fR = cv2.resize(fR, self.size_single)
        fU = cv2.resize(fU, self.size_single)
        canvas = np.hstack([fR, fU])

        # 텍스트 크기 자동 계산
        h, w = canvas.shape[:2]
        font_scale = h / 300.0
        thickness = int(font_scale * 2)
        size = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        org = (w//2 - size[0]//2, h//2 + size[1]//2)

        # 카운트다운 텍스트 (그림자+본문)
        cv2.putText(canvas, txt, org, cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0,0,0), thickness+2, cv2.LINE_AA)
        cv2.putText(canvas, txt, org, cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255,255,255), thickness, cv2.LINE_AA)

        self.update_canvas(canvas)

    def start_video(self):
        self.timer.start(30)

    def update_frame(self):
        okR, fR = self.capR.read()
        okU, fU = self.capU.read()
        if not okR or not okU:
            self.timer.stop()
            return

        # 프레임 리사이즈
        fR = cv2.resize(fR, self.size_single)
        fU = cv2.resize(fU, self.size_single)
        self.frame_idx += 1
        do_score = (self.frame_idx % self.args.every == 0)

        # 1) 포즈 추론 + 스코어링
        if do_score:
            try:
                kpsR, confR = self.infer_pose(fR)
                kpsU, confU = self.infer_pose(fU)
            except Exception as e:
                print(f"[warn] 추론 실패:", e)
                return

            if kpsR is not None:
                self.lastR = {"kps": kpsR, "conf": confR}
            if kpsU is not None:
                self.lastU = {"kps": kpsU, "conf": confU}

            if self.lastR["kps"] is not None and self.lastU["kps"] is not None:
                KR_n = normalize_keypoints(self.lastR["kps"])
                KU_use = (flip_horizontal_pts(self.lastU["kps"])
                          if self.mirror else self.lastU["kps"])
                KU_n = normalize_keypoints(KU_use)
                vR = pose_to_anglevec(KR_n)
                vU = pose_to_anglevec(KU_n)
                s, _, _ = frame_score_strict(vR, vU)

                # EMA 업데이트
                self.ema_score = s if self.ema_score is None else (
                    self.ema_score*(1-self.ema_alpha) + s*self.ema_alpha)

                # total_points & 상태 결정
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

                print(f"[total]@{self.total_points}")

        # 2) 포즈 그리기
        if self.lastR["kps"] is not None and self.lastR["kps"].size > 0:
            draw_pose(fR, self.lastR["kps"], kps_conf=self.lastR["conf"])
        if self.lastU["kps"] is not None and self.lastU["kps"].size > 0:
            draw_pose(fU, self.lastU["kps"], kps_conf=self.lastU["conf"])

        # 3) 캔버스 생성
        canvas = np.hstack([fR, fU])

        # 4) 스코어 텍스트 (상단 중앙)
        if self.timer.isActive() and self.ema_score is not None:
            ch, cw = canvas.shape[:2]
            score_txt = f"Score: {self.ema_score:.1f}"
            scale = ch / 400.0
            thick = int(scale * 2)
            sz = cv2.getTextSize(score_txt, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)[0]
            org = (cw//2 - sz[0]//2, int(ch * 0.08))
            cv2.putText(canvas, score_txt, org, cv2.FONT_HERSHEY_SIMPLEX,
                        scale, (0,0,0), thick+2, cv2.LINE_AA)
            cv2.putText(canvas, score_txt, org, cv2.FONT_HERSHEY_SIMPLEX,
                        scale, (255,255,255), thick, cv2.LINE_AA)

            # 상태 텍스트 (하단 중앙)
            if self.status_text:
                ch, cw = canvas.shape[:2]

                # 상태 텍스트는 점수보다 크게
                scale = ch / 250.0   # 기존보다 크게
                thick = int(scale * 3)

                # 텍스트 크기 계산
                sz = cv2.getTextSize(self.status_text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)[0]
                org = (cw // 2 - sz[0] // 2, int(ch * 0.90))  # 하단 중앙

                # 그림자 (검은색, 두껍게)
                cv2.putText(canvas, self.status_text, org, cv2.FONT_HERSHEY_SIMPLEX,
                                        scale, (0, 0, 0), thick + 4, cv2.LINE_AA)

                # 본문 (Good=초록, Bad=빨강, Neutral=회색)
                cv2.putText(canvas, self.status_text, org, cv2.FONT_HERSHEY_SIMPLEX,
                            scale, self.status_color, thick, cv2.LINE_AA)

        # 6) 화면에 업데이트
        self.update_canvas(canvas)

        # 7) 파일 저장
        if self.writer and self.writer.isOpened():
            self.writer.write(canvas)

    def update_canvas(self, canvas):
        self.last_canvas = canvas.copy()
        vw, vh = self.graphicsView.viewport().size().width(), \
                  self.graphicsView.viewport().size().height()
        resized = cv2.resize(canvas, (vw, vh), interpolation=cv2.INTER_AREA)
        h, w = resized.shape[:2]
        qimg = QImage(resized.data, w, h, 3*w, QImage.Format_RGB888).rgbSwapped()
        pix = QPixmap.fromImage(qimg)
        self.scene.clear()
        self.scene.setSceneRect(0, 0, w, h)
        self.scene.addPixmap(pix)
        self.graphicsView.setScene(self.scene)

    def resizeEvent(self, ev):
        if self.last_canvas is not None:
            self.update_canvas(self.last_canvas)
        super().resizeEvent(ev)

    def closeEvent(self, ev):
        if self.capR: self.capR.release()
        if self.capU: self.capU.release()
        if self.writer: self.writer.release()
        print(f"[final] total score = {self.total_points}")
        ev.accept()
