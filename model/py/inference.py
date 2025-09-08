
import os
import numpy as np
import torch
import cv2
from tqdm import tqdm
from settings import DETECT_CONF_THRES, KPT_CONF_THRES

# ---------------- live 추론(트래킹) ----------------
def track_once(model, frame, tracker_yaml, imgsz, device, half, max_persons=None):
    with torch.inference_mode():
        results = model.track(frame,imgsz=imgsz,device=device,half=half,
                              conf=DETECT_CONF_THRES,verbose=False,
                              persist=True,tracker=tracker_yaml,stream=False)
    if not results: return {}
    res=results[0]
    if (res.keypoints is None) or (len(res.keypoints)==0): return {}
    ids=getattr(res.boxes,"id",None)
    if ids is None: return {}
    ids=ids.detach().cpu().numpy().astype(int)
    boxes=res.boxes.xyxy.detach().cpu().numpy()
    out={}
    for i in range(len(ids)):
        tid=int(ids[i])
        kps=res.keypoints.xy[i].detach().cpu().numpy()
        conf=res.keypoints.conf[i].detach().cpu().numpy()
        kps[conf<KPT_CONF_THRES]=np.nan
        box=boxes[i].astype(float)
        out[tid]=(kps,conf,box)
    
    if max_persons is not None and len(out) > max_persons:
        # Sort by track ID and take the top 'max_persons'
        sorted_tids = sorted(out.keys())[:max_persons]
        out = {tid: out[tid] for tid in sorted_tids}
        
    return out

# ---------------- ref 전처리 (다인원 지원) ----------------
def preprocess_reference_tracks(ref_path, model, tracker_yaml, imgsz, device, half, force_preprocess=False):
    cache_path = ref_path + ".npz"
    if not force_preprocess and os.path.exists(cache_path):
        try:
            with np.load(cache_path, allow_pickle=True) as data:
                ref_tracks_all_frames = data['tracks']
            print(f"[info] Loaded {len(ref_tracks_all_frames)} pre-processed frames from cache.")
            return ref_tracks_all_frames
        except Exception as e:
            print(f"[warn] Cache loading failed: {e}. Re-processing...")

    cap = cv2.VideoCapture(ref_path)
    if not cap.isOpened(): raise RuntimeError(f"Cannot open reference: {ref_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ref_tracks_all_frames = []
    
    print("[info] Pre-processing reference video...")
    for _ in tqdm(range(total_frames), desc="Processing reference frames"):
        ok, frame = cap.read()
        if not ok: break
        tracks = track_once(model, frame, tracker_yaml, imgsz, device, half)
        ref_tracks_all_frames.append(tracks)
    
    cap.release()
    np.savez_compressed(cache_path, tracks=np.array(ref_tracks_all_frames, dtype=object))
    print(f"[info] Pre-processing finished. Saved {len(ref_tracks_all_frames)} frames to cache.")
    return np.array(ref_tracks_all_frames, dtype=object)
