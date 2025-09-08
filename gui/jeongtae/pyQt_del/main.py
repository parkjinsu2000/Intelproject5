# -*- coding: utf-8 -*-
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap

# from py_code import realtime_pose_score  # 같은 폴더에 있어야 함
from realtime_pose_score import PoseWorker   # 또는 from py_code.pose_worker import PoseWorker

from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot

# ---- 환경 값 ----
MODEL_PATH = "yolov8l-pose.pt"
REF_PATH   = "ref_1.mp4"
CAM_INDEX  = 0
START_SEC  = 3.0
MIRROR     = True
# --------------

class DummyWorker(QThread):
    frame_ready = pyqtSignal(object)
    info_ready  = pyqtSignal(str)
    def run(self):
        import numpy as np, cv2, time
        self.info_ready.emit("[info] dummy worker running")
        for i in range(120):
            img = np.zeros((360, 640, 3), np.uint8)
            cv2.putText(img, f"frame {i}", (40,100), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,255,255),3)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            from PyQt5.QtGui import QImage
            qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[1]*3, QImage.Format_RGB888).copy()
            self.frame_ready.emit(qimg)
            time.sleep(0.03)
        self.info_ready.emit("[info] dummy worker done")

class MainWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pose Compare (PyQt5)")
        self.resize(1280, 720)

        self.video_label = QLabel("영상 출력 영역")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background:#202020; color:#AAAAAA;")
        self.video_label.setMinimumSize(960, 360)
        self.video_label.setScaledContents(True)

        self.btn = QPushButton("시작")
        self.btn.clicked.connect(self.on_toggle)

        layout = QVBoxLayout(self)
        layout.addWidget(self.video_label)
        layout.addWidget(self.btn)

        self.worker = None

    def on_toggle(self):
        if self.worker and self.worker.isRunning():
            self.btn.setEnabled(False)
            self.worker.stop()
            self.worker.wait()
            self.btn.setText("시작")
            self.btn.setEnabled(True)
            return

        # 시작
        self.worker = PoseWorker(
            model_path=MODEL_PATH,
            ref_path=REF_PATH,
            cam_index=CAM_INDEX,
            mirror=MIRROR,
            start_sec=START_SEC,
        )
        # self.worker = DummyWorker()
        self.worker.frame_ready.connect(self.update_frame)
        self.worker.info_ready.connect(self.on_info)
        self.worker.start()
        self.btn.setText("정지")

    def update_frame(self, qimg):
        self.video_label.setPixmap(QPixmap.fromImage(qimg))

    def on_info(self, msg: str):
        self.setWindowTitle(f"Pose Compare (PyQt5)  {msg}")

    def closeEvent(self, e):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
        super().closeEvent(e)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWidget()
    w.show()
    sys.exit(app.exec_())

