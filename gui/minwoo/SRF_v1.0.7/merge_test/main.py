import torch
from ultralytics import YOLO
from PyQt5.QtWidgets import QApplication
from pages.main_window import MainWindow

def main():
    # 모델 로드
    model = YOLO("yolov8l-pose.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    try:
        model.fuse()
    except:
        pass

    use_half = (device == "cuda")
    if use_half:
        try:
            model.model.half()
        except:
            use_half = False

    # Qt 앱 실행
    app = QApplication([])
    window = MainWindow(model, use_half)
    window.show()
    app.exec_()

if __name__ == "__main__":
    main()
