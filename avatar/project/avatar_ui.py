# avatar_ui.py
import os
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import os as _os, cv2, sys
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QGroupBox,
                             QRadioButton, QPushButton, QProgressBar, QLabel)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSlot   # ← pyqtSlot 추가
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
    def start_render(self, json_path, assets_dir):
        # 실행 중이면 먼저 멈춤
        self.stop_render()

        from PyQt5.QtCore import QThread
        self.thread = QThread(self)                  # 부모는 윈도우(수명 관리)
        self.renderer = MannequinRenderer(json_path, assets_dir)  # parent=None 권장
        self.renderer.moveToThread(self.thread)

        # 시작/종료 배선
        self.thread.started.connect(self.renderer.run)
        # MannequinRenderer에 finished 신호가 있다고 가정
        self.renderer.finished.connect(self.thread.quit)           # 끝나면 스레드 종료
        self.renderer.finished.connect(self.renderer.deleteLater)  # 워커 정리
        self.thread.finished.connect(self.thread.deleteLater)      # 스레드 정리

        # 진행/로그/결과 연결 (이 이름의 슬롯을 아래에 추가함)
        self.renderer.progress.connect(self.on_progress)
        self.renderer.log.connect(self.on_log)
        self.renderer.error.connect(self.on_error)
        self.renderer.playReady.connect(self.on_play_ready)

        # 진행바 초기화
        self.progress.setValue(0)
        self.progress.show()

        self.thread.start()

    def stop_render(self, wait_ms=3000):
        # 렌더 중이면 취소→종료→대기
        if hasattr(self, "renderer") and self.renderer:
            try:
                self.renderer.cancel()  # 루프가 자연 종료되도록 (있다면)
            except Exception:
                pass
        if hasattr(self, "thread") and self.thread:
            if self.thread.isRunning():
                self.thread.quit()
                self.thread.wait(wait_ms)  # 반드시 대기

    def closeEvent(self, event):
        # 창 닫을 때 스레드 정리
        self.stop_render()
        super().closeEvent(event)

    # ---------- 기존 내부 콜백 ----------
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

    # ---------- 신호에 연결될 슬롯들(추가) ----------
    @pyqtSlot(int)
    def on_progress(self, value: int):
        try:
            if self.progress.isHidden():
                self.progress.show()
            self.progress.setValue(int(value))
            if value >= 100:
                self.progress.hide()
        except Exception:
            pass

    @pyqtSlot(str)
    def on_log(self, text: str):
        # 필요 시 별도 로그 위젯로 보내세요. 일단 콘솔 출력.
        print(text)

    @pyqtSlot(str)
    def on_error(self, msg: str):
        self._on_error(msg)

    @pyqtSlot(object, float)
    def on_play_ready(self, frames, fps: float):
        self._on_ready(frames, fps)

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
    2: ("sodapop.json", "rumi_parts"),
    3: ("sodapop.json", "dady_parts"),
    4: ("dance_poses.json", "ren_parts"),
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
