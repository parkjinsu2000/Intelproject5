import cv2
import numpy as np
from PyQt5.QtWidgets import QMainWindow, QGraphicsView, QGraphicsScene, QPushButton, QMessageBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer, Qt
from pose_utils import (
    normalize_keypoints,
    flip_horizontal_pts,
    pose_to_anglevec,
    frame_score_strict,
    draw_pose,
    put_text
)

from model_loader import make_infer

class PoseScoreApp(QMainWindow):
    def __init__(self, args, model, use_half):
        super().__init__()
        self.args = args
        self.model = model
        self.use_half = use_half
        self.infer_pose = make_infer(model, args, use_half)

        self.total_points = 100
        self.ema_score = None
        self.ema_alpha = 0.25
        self.frame_idx = 0
        self.lastR = {"kps": None, "conf": None}
        self.lastU = {"kps": None, "conf": None}
        self.mirror = not self.args.no_mirror

        self.init_ui()
        self.init_video()

    def init_ui(self):
        self.setWindowTitle("Pose Score Viewer")
        self.resize(1280, 720)

        self.scene = QGraphicsScene(self)
        self.graphicsView = QGraphicsView(self.scene, self)
        self.graphicsView.setGeometry(10, 60, 1260, 640)

        self.button = QPushButton("영상 시작", self)
        self.button.setGeometry(10, 10, 200, 40)
        self.button.clicked.connect(self.start_video)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def init_video(self):
        self.capR = cv2.VideoCapture(self.args.ref)
        self.capU = cv2.VideoCapture(self.args.cam)

        if not self.capR.isOpened():
            QMessageBox.critical(self, "오류", f"기준 영상 열 수 없음: {self.args.ref}")
            return
        if not self.capU.isOpened():
            QMessageBox.critical(self, "오류", "카메라를 열 수 없습니다.")
            return

        W = int(max(self.capR.get(cv2.CAP_PROP_FRAME_WIDTH), 640))
        H = int(max(self.capR.get(cv2.CAP_PROP_FRAME_HEIGHT), 360))
        self.size_single = (W, H)

        self.writer = None
        if self.args.save:
            fps = self.capR.get(cv2.CAP_PROP_FPS) or 30.0
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(self.args.save, fourcc, int(round(fps)), (W * 2, H))

    def start_video(self):
        self.timer.start(30)

    def update_frame(self):
        okR, fR = self.capR.read()
        okU, fU = self.capU.read()
        if not okR or not okU:
            return

        fR = cv2.resize(fR, self.size_single)
        fU = cv2.resize(fU, self.size_single)
        self.frame_idx += 1
        do_score = (self.frame_idx % self.args.every == 0)

        if do_score:
            try:
                kpsR, confR = self.infer_pose(fR)
                kpsU, confU = self.infer_pose(fU)
            except Exception as e:
                print(f"[warn] 추론 실패: {e}")
                return

            if kpsR is not None: self.lastR = {"kps": kpsR, "conf": confR}
            if kpsU is not None: self.lastU = {"kps": kpsU, "conf": confU}

            if self.lastR["kps"] is not None and self.lastU["kps"] is not None:
                kR_n = normalize_keypoints(self.lastR["kps"])
                kU_use = flip_horizontal_pts(self.lastU["kps"]) if self.mirror else self.lastU["kps"]
                kU_n = normalize_keypoints(kU_use)
                vecR = pose_to_anglevec(kR_n)
                vecU = pose_to_anglevec(kU_n)
                s, _, _ = frame_score_strict(vecR, vecU)
                self.ema_score = s if self.ema_score is None else (self.ema_score*(1-self.ema_alpha) + s*self.ema_alpha)
                if s < 50.0:
                    self.total_points = max(0, self.total_points - 1)
                elif s >= 70.0:
                    self.total_points = min(100, self.total_points + 1)
                print(f"[total]@{self.total_points}")


                if self.lastR["kps"] is not None and self.lastR["kps"].size > 0:
                    draw_pose(fR, self.lastR["kps"], kps_conf=self.lastR["conf"])

                if self.lastU["kps"] is not None and self.lastU["kps"].size > 0:
                    draw_pose(fU, self.lastU["kps"], kps_conf=self.lastU["conf"])

        canvas = np.hstack([fR, fU])
        self.update_canvas(canvas)

        if self.writer and self.writer.isOpened():
            self.writer.write(canvas)

    def update_canvas(self, canvas):
        scale = self.args.disp_scale
        max_width = self.args.disp_width
        if max_width and max_width > 0:
            scale = min(scale, float(max_width) / canvas.shape[1])
        if scale < 1.0:
            canvas = cv2.resize(canvas, (int(canvas.shape[1]*scale), int(canvas.shape[0]*scale)), interpolation=cv2.INTER_AREA)

        h, w, ch = canvas.shape
        qimg = QImage(canvas.data, w, h, 3*w, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(qimg)
        self.scene.clear()
        self.scene.addPixmap(pixmap)
        self.graphicsView.setScene(self.scene)
        self.graphicsView.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)

    def closeEvent(self, event):
        if self.capR: self.capR.release()
        if self.capU: self.capU.release()
        if self.writer: self.writer.release()
        cv2.destroyAllWindows()
        print(f"[final] total score = {self.total_points}")
        event.accept()
