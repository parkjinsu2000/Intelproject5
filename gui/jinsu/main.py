import sys
import torch
from ultralytics import YOLO
from PyQt5.QtCore import Qt, QCoreApplication, QEvent, QTimer
from PyQt5.QtWidgets import QApplication, QMainWindow, QStackedWidget, QSizePolicy
from video_select import VideoSelectPage

class MainWindow(QMainWindow):
    def __init__(self, model, use_half):
        super().__init__()
        self.setWindowTitle("Pose Score App")
        self.resize(1280, 720)

        self.stacked = QStackedWidget()
        self.stacked.setContentsMargins(0, 0, 0, 0)
        self.setCentralWidget(self.stacked)

        select_page = VideoSelectPage(self.stacked, model, use_half)
        select_page.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        if select_page.layout():
            select_page.layout().setContentsMargins(0, 0, 0, 0)
            select_page.layout().setSpacing(0)

        self.stacked.addWidget(select_page)
        self.stacked.setCurrentWidget(select_page)

    def changeEvent(self, event):
        if event.type() == QEvent.WindowStateChange:
            page = self.stacked.currentWidget()
            if hasattr(page, "equalize_splitter"):
                QTimer.singleShot(0, page.equalize_splitter)
        super().changeEvent(event)

    def resizeEvent(self, event):
        page = self.stacked.currentWidget()
        if hasattr(page, "equalize_splitter"):
            QTimer.singleShot(0, page.equalize_splitter)
        super().resizeEvent(event)

def main():
    QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QCoreApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    model = YOLO("yolov8l-pose.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    try:
        model.fuse()
    except:
        pass

    use_half = device == "cuda"
    if use_half:
        try:
            model.model.half()
        except:
            use_half = False

    dummy = torch.zeros((1, 3, 384, 384), dtype=torch.float32)
    with torch.inference_mode():
        _ = model.predict(dummy, imgsz=320, device=device, half=use_half, conf=0.25)[0]

    app = QApplication(sys.argv)
    window = MainWindow(model, use_half)
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
