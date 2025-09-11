import torch
from ultralytics import YOLO
from types import SimpleNamespace
from pose_score_app import PoseScoreApp
from PyQt5.QtWidgets import QApplication, QStackedWidget
from video_select import VideoSelectPage

def main():
    model = YOLO("yolov8l-pose.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    try: model.fuse()
    except: pass

    use_half = device == "cuda"
    if use_half:
        try: model.model.half()
        except: use_half = False

    dummy = torch.zeros((1, 3, 384, 384), dtype=torch.float32)
    with torch.inference_mode():
        _ = model.predict(dummy, imgsz=320, device=device, half=use_half, conf=0.25)[0]

    app = QApplication([])
    stacked = QStackedWidget()
    select_page = VideoSelectPage(stacked, model, use_half)
    stacked.addWidget(select_page)
    stacked.setCurrentWidget(select_page)
    stacked.setMinimumSize(800, 600)  # 최소 크기만 제한
    stacked.show()
    app.exec_()

# 👇 이 부분은 반드시 함수 밖에 있어야 합니다!
if __name__ == "__main__":
    main()
