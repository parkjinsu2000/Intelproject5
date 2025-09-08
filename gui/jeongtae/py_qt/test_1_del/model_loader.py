# model_loader.py
import torch
import numpy as np
from ultralytics import YOLO
from pose_utils import KPT_CONF_THRES

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
