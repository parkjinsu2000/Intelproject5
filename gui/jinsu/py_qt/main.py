import torch
from ultralytics import YOLO
from types import SimpleNamespace
from widget import PoseScoreApp
from PyQt5.QtWidgets import QApplication

def main():
    args = SimpleNamespace(
        ref="person.mp4",
        cam=0,
        start=3.0,
        every=5,
        no_mirror=False,
        disp_scale=1.0,
        disp_width=1280,
        save="output.mp4",
        imgsz=320,
        device="cuda" if torch.cuda.is_available() else "cpu",
        half=True,
        conf_thres=0.25
    )

    model = YOLO("yolov8l-pose.pt")
    model.to(args.device)
    try: model.fuse()
    except: pass
    use_half = args.half and args.device == "cuda"
    if use_half:
        try: model.model.half()
        except: use_half = False

    dummy = torch.zeros((1, 3, 384, 384), dtype=torch.float32)
    with torch.inference_mode():
        _ = model.predict(dummy, imgsz=args.imgsz, device=args.device, half=use_half, conf=args.conf_thres)[0]

    app = QApplication([])
    window = PoseScoreApp(args, model, use_half)
    window.show()
    app.exec_()

if __name__ == "__main__":
    main()
