# avatar_ui.py
import os
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import os as _os, cv2, sys
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QGroupBox,
                             QRadioButton, QPushButton, QProgressBar, QLabel)
from PyQt5.QtCore import Qt, QTimer, QThread
from PyQt5.QtGui import QPixmap, QImage, QPainter
from avatar_qt import MannequinRenderer

# 배경 이미지 경로 (이 파일과 같은 폴더에 SRF.png 가 있다고 가정)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BG_IMAGE = os.path.join(_THIS_DIR, "SRF.png")


class CoverBgWidget(QWidget):
    """창 배경을 'cover' 방식으로 꽉 채워 그리는 위젯 베이스"""
    def __init__(self, bg_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._bg_pm = QPixmap(bg_path)
        self.setAutoFillBackground(False)

    def paintEvent(self, e):
        p = QPainter(self)
        if self._bg_pm.isNull():
            p.fillRect(self.rect(), Qt.black)
            return
        w, h = self.width(), self.height()
        pm = self._bg_pm.scaled(w, h, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
        x = (w - pm.width()) // 2
        y = (h - pm.height()) // 2
        p.drawPixmap(x, y, pm)
        # super().paintEvent(e)  # 배경 위에 그릴 필요 없어서 생략


class Viewer(CoverBgWidget):
    def __init__(self):
        super().__init__(BG_IMAGE)
        self.setWindowTitle("Avatar Viewer")
        self.resize(2560, 1440)

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)   # 여백 제거
        v.setSpacing(0)

        self.view = QLabel("", self)
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setStyleSheet("background: transparent;")  # 배경 비치게
        self.progress = QProgressBar(self)
        self.progress.setTextVisible(False)
        self.progress.setRange(0, 100)
        self.progress.hide()

        v.addWidget(self.view, 1)
        v.addWidget(self.progress)

        # ProgressBar 스타일
        self.setStyleSheet("""
QProgressBar {
    background: rgba(0,0,0,120);
    color: white;
    border: 1px solid rgba(255,255,255,60);
    border-radius: 8px;
    height: 26px;
}
QProgressBar::chunk {
    background: rgba(255,255,255,180);
    border-radius: 6px;
}
""")

        # 재생 상태
        self.frames = []
        self.fps = 24.0
        self.idx = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._next)

        # MP4 프리뷰
        self.preview_cap = None
        self.preview_timer = QTimer(self)
        self.preview_timer.timeout.connect(self._next_preview)
        self.preview_interval_ms = 40
        self.preview_qimg = None

        self.thread = None
        self.renderer = None

    # ---------- MP4 프리뷰 ----------
    def _start_preview(self, mp4_path: str):
        self._stop_preview()
        cap = cv2.VideoCapture(mp4_path)
        if not cap.isOpened():
            self.view.setText(f"로딩 중... (프리뷰 열기 실패)\n{mp4_path}")
            return
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.preview_interval_ms = max(1, int(round(1000.0 / fps)))
        self.preview_cap = cap
        ok, frame = cap.read()
        if ok:
            self._show_preview_frame(frame)
            self.preview_timer.start(self.preview_interval_ms)
        else:
            self.view.setText(f"로딩 중... (프리뷰 프레임 없음)\n{mp4_path}")
            self._stop_preview()

    def _next_preview(self):
        if not self.preview_cap:
            return
        ok, frame = self.preview_cap.read()
        if not ok:
            self.preview_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.preview_cap.read()
            if not ok:
                self._stop_preview()
                return
        self._show_preview_frame(frame)

    def _show_preview_frame(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = frame_rgb.shape
        qimg = QImage(frame_rgb.data, w, h, frame_rgb.strides[0], QImage.Format_RGB888).copy()
        self.preview_qimg = qimg
        pix = QPixmap.fromImage(qimg).scaled(
            self.view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.view.setPixmap(pix)

    def _stop_preview(self):
        if self.preview_timer.isActive():
            self.preview_timer.stop()
        if self.preview_cap:
            try: self.preview_cap.release()
            except Exception: pass
        self.preview_cap = None
        self.preview_qimg = None

    # ---------- 결과 프레임 재생 ----------
    def _show(self, i):
        pix = QPixmap.fromImage(self.frames[i]).scaled(
            self.view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.view.setPixmap(pix)

    def _next(self):
        if not self.frames:
            return
        self.idx = (self.idx + 1) % len(self.frames)
        self._show(self.idx)

    # ---------- 렌더링 제어 ----------
    def start_render(self, json_path, assets_dir, show_debug=True):
        self.stop_render()
        self.progress.setValue(0)
        self.progress.show()
        # self.view.setText("로딩 중...")

        mp4_path = _os.path.splitext(json_path)[0] + ".mp4"
        if _os.path.exists(mp4_path):
            self._start_preview(mp4_path)
        else:
            self.view.setText(f"로딩 중... (프리뷰 없음)\n{mp4_path}")

        self.renderer = MannequinRenderer(
            json_path=json_path,
            assets_dir=assets_dir,
            show_debug=False,  # 스켈레톤 점 숨김
            parent=None
        )
        self.thread = QThread(self)
        self.renderer.moveToThread(self.thread)

        self.renderer.progress.connect(self.progress.setValue)
        self.renderer.error.connect(self._on_error)
        self.renderer.playReady.connect(self._on_ready)

        self.thread.started.connect(self.renderer.run)
        self.thread.start()

    def stop_render(self):
        self.timer.stop()
        self._stop_preview()
        if self.thread:
            self.thread.quit()
            self.thread.wait(3000)
        self.thread = None
        self.renderer = None
        self.frames = []

    def _on_error(self, msg):
        self.progress.hide()
        self.view.setText("ERROR: " + msg)
        self.stop_render()

    def _on_ready(self, qframes, fps):
        self._stop_preview()
        self.frames = qframes or []
        self.fps = max(1.0, float(fps))
        self.idx = 0
        self.progress.hide()
        if not self.frames:
            self.view.setText("프레임 없음")
            return
        self._show(0)
        self.timer.start(int(round(1000.0 / self.fps)))

    def resizeEvent(self, e):
        # 배경은 paintEvent에서 자동 리렌더링
        if self.preview_qimg is not None:
            pix = QPixmap.fromImage(self.preview_qimg).scaled(
                self.view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.view.setPixmap(pix)
        elif self.frames:
            self._show(self.idx)
        super().resizeEvent(e)


class Controller(CoverBgWidget):
    def __init__(self, viewer: Viewer, options: dict):
        super().__init__(BG_IMAGE)
        self.viewer = viewer
        self.options = options
        self.setWindowTitle("변환 제어판")
        self.resize(1920, 1080)

        # 반투명 패널 스타일
        self.setStyleSheet("""
QGroupBox {
    background-color: rgba(0,0,0,120);
    color: white;
    border: 1px solid rgba(255,255,255,60);
    border-radius: 12px;
    margin-top: 24px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
}
QRadioButton, QPushButton {
    color: white;
    font-size: 16px;
}
QPushButton {
    background-color: rgba(255,255,255,40);
    border: 1px solid rgba(255,255,255,80);
    border-radius: 8px;
    padding: 8px 16px;
}
QPushButton:hover {
    background-color: rgba(255,255,255,70);
}
""")

        v = QVBoxLayout(self)
        v.setContentsMargins(24, 24, 24, 24)
        v.setSpacing(16)

        gb = QGroupBox("")
        gg = QVBoxLayout(gb)
        gg.setContentsMargins(16, 16, 16, 16)
        gg.setSpacing(8)
        self.rb = []
        for i in range(1, 1 + len(options)):
            r = QRadioButton(f"{i}번")
            gg.addWidget(r)
            self.rb.append(r)
        v.addWidget(gb)

        self.btn = QPushButton("변환하기")
        self.btn.clicked.connect(self._convert)
        v.addWidget(self.btn, alignment=Qt.AlignRight)

    def _convert(self):
        sel = None
        for idx, r in enumerate(self.rb, start=1):
            if r.isChecked():
                sel = idx
                break
        if sel is None:
            # self.viewer.view.setText("먼저 세트를 선택하세요.")
            return
        json_path, assets_dir = self.options.get(sel, list(self.options.values())[0])
        self.viewer.start_render(json_path, assets_dir)


DEFAULT_OPTIONS = {
    1: ("dance_poses.json", "naruto_parts"),
    2: ("dance_poses.json", "mannequin_parts"),
    3: ("dance_poses.json", "naruto_parts_alt"),
    4: ("dance_poses.json", "naruto_parts"),
}

def run_app(options: dict = None):
    """다른 프로젝트에서도 바로 호출 가능한 진입점"""
    app = QApplication(sys.argv)

    # 옵션이 안 들어오면 기본 옵션 사용
    opts = options or DEFAULT_OPTIONS

    viewer = Viewer()
    viewer.move(520, 100)
    viewer.show()

    ctrl = Controller(viewer, opts)
    ctrl.move(100, 150)
    ctrl.show()

    sys.exit(app.exec_())
