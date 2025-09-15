import torch
import numpy as np
from ultralytics import YOLO
from core.pose_utils import KPT_CONF_THRES

def load_model(model_path: str, device: str, use_half: bool):
    """
    Loads a YOLO pose model from the given path.
    """
    if not torch.cuda.is_available() and device == 'cuda':
        print("CUDA is not available. Using CPU instead.")
        device = 'cpu'
        use_half = False
    
    try:
        model = YOLO(model_path)
        model.to(device)
        model.eval()
        print(f"Successfully loaded model from {model_path} on device {device}.")
        return model, use_half
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, use_half

def make_infer(model, args, use_half: bool):
    def infer_pose(frame):
        with torch.inference_mode():
            res = model.predict(
                frame, imgsz=args.imgsz, device=args.device,
                half=use_half, conf=args.conf_thres, verbose=False
            )[0]
        if (res.keypoints is None) or (len(res.keypoints) == 0):
            return None, None
        if len(res.boxes) > 1:
            areas = (res.boxes.xywh[:,2] * res.boxes.xywh[:,3]).detach().cpu().numpy()
            idx = int(np.argmax(areas))
        else:
            idx = 0
        kps = res.keypoints.xy[idx].detach().cpu().numpy()
        conf = res.keypoints.conf[idx].detach().cpu().numpy()
        kps[conf < KPT_CONF_THRES] = np.nan
        return kps, conf
    return infer_pose
