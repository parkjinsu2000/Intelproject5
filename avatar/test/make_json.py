#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_pose_json.py
- JT.mp4에서 프레임별 COCO 17점 포즈를 추출해 dance_poses.json으로 저장
- 좌표는 '원본 비디오 픽셀 단위' (normalized=False)
- 가장 큰 사람 1명만 추출, 신뢰도 낮은 키포인트는 null 처리
- STRIDE 로 N프레임마다 샘플링
- 추가: 저장 기능
  * 원본 위 오버레이 미리보기 MP4 (poses_preview.mp4)
  * 검은 배경 스켈레톤만 MP4 (json_skeleton.mp4)
"""

import json, os
import numpy as np
import cv2
from ultralytics import YOLO

# ---------- 설정 ----------
VIDEO_PATH = "naruto.mp4"                 # 입력 영상
OUT_JSON   = "naruto.json"       # 출력 json 파일
MODEL_PATH = "yolov8l-pose.pt"        # 요구 모델

DETECT_CONF_THRES = 0.25              # 사람 감지 임계
KPT_CONF_THRES    = 0.20              # 키포인트 임계(이하이면 null)
STRIDE            = 3                 # 5로 두면 '5프레임마다' 추출

# 저장 옵션
SAVE_PREVIEW_MP4   = True             # 원본 위 포즈 오버레이
PREVIEW_PATH       = "poses_preview.mp4"

SAVE_SKELETON_MP4  = True             # 검은 배경 스켈레톤만
SKELETON_PATH      = "json_skeleton.mp4"
# --------------------------

# COCO keypoint indices
NOSE=0; L_EYE=1; R_EYE=2; L_EAR=3; R_EAR=4
L_SH=5; R_SH=6; L_EL=7; R_EL=8; L_WR=9; R_WR=10
L_HP=11; R_HP=12; L_KN=13; R_KN=14; L_AN=15; R_AN=16

EDGES = [
    (L_SH,L_EL),(L_EL,L_WR),(R_SH,R_EL),(R_EL,R_WR),
    (L_HP,L_KN),(L_KN,L_AN),(R_HP,R_KN),(R_KN,R_AN),
    (L_SH,R_SH),(L_HP,R_HP),(L_SH,L_HP),(R_SH,R_HP),
    (NOSE,L_SH),(NOSE,R_SH)
]

def get_main_pose(frame, model):
    """가장 큰 사람의 (kps_xy[17,2], kps_conf[17]) 반환. 없으면 (None,None)"""
    res = model(frame, conf=DETECT_CONF_THRES, verbose=False)[0]
    if (res.keypoints is None) or (len(res.keypoints) == 0):
        return None, None
    # 가장 큰 박스 선택
    if len(res.boxes) > 1:
        areas = (res.boxes.xywh[:,2] * res.boxes.xywh[:,3]).cpu().numpy()
        idx = int(np.argmax(areas))
    else:
        idx = 0
    kps  = res.keypoints.xy[idx].cpu().numpy()    # (17,2)
    conf = res.keypoints.conf[idx].cpu().numpy()  # (17,)
    # 저신뢰는 NaN 좌표로
    kps[conf < KPT_CONF_THRES] = np.nan
    return kps, conf

def draw_pose_overlay(img, kps):
    """원본 프레임 위에 포즈 그리기(오버레이)"""
    if kps is None: return img
    vis = img.copy()
    for i,j in EDGES:
        pi, pj = kps[i], kps[j]
        if np.all(np.isfinite(pi)) and np.all(np.isfinite(pj)):
            cv2.line(vis, tuple(np.round(pi).astype(int)), tuple(np.round(pj).astype(int)),
                     (0,255,0), 2, cv2.LINE_AA)
    for p in kps:
        if np.all(np.isfinite(p)):
            cv2.circle(vis, tuple(np.round(p).astype(int)), 3, (255,255,255), -1, cv2.LINE_AA)
    return vis

def draw_skeleton_black(kps, W, H):
    """검은 배경에 스켈레톤만 그리기"""
    img = np.zeros((H, W, 3), np.uint8)
    if kps is None: return img
    for i,j in EDGES:
        pi, pj = kps[i], kps[j]
        if np.all(np.isfinite(pi)) and np.all(np.isfinite(pj)):
            cv2.line(img, tuple(np.round(pi).astype(int)), tuple(np.round(pj).astype(int)),
                     (0,255,0), 2, cv2.LINE_AA)
    for p in kps:
        if np.all(np.isfinite(p)):
            cv2.circle(img, tuple(np.round(p).astype(int)), 3, (255,255,255), -1, cv2.LINE_AA)
    return img

def main():
    if not os.path.isfile(VIDEO_PATH):
        raise FileNotFoundError(f"video not found: {VIDEO_PATH}")

    print("[info] loading model:", MODEL_PATH)
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    N   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 미리보기/스켈레톤 영상 저장기
    writer_preview  = None
    writer_skeleton = None
    out_fps = max(1.0, fps / max(1, STRIDE))   # 샘플링된 프레임 속도에 맞춰 저장
    if SAVE_PREVIEW_MP4:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer_preview = cv2.VideoWriter(PREVIEW_PATH, fourcc, out_fps, (W, H))
    if SAVE_SKELETON_MP4:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer_skeleton = cv2.VideoWriter(SKELETON_PATH, fourcc, out_fps, (W, H))

    frames_out = []
    frame_idx = 0
    grabbed = 0
    print(f"[info] video size: {W}x{H}, fps={fps:.3f}, frames={N}, stride={STRIDE}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % STRIDE != 0:
            frame_idx += 1
            continue

        kps, conf = get_main_pose(frame, model)  # kps: (17,2) with NaN
        if kps is None:
            kps_list = [[None, None] for _ in range(17)]
        else:
            kps_list = [[None if not np.isfinite(x) else float(x),
                         None if not np.isfinite(y) else float(y)]
                        for (x,y) in kps]

        frames_out.append({"kps": kps_list})

        # 저장: 오버레이
        if writer_preview is not None:
            vis = draw_pose_overlay(frame, kps)
            writer_preview.write(vis)

        # 저장: 스켈레톤만
        if writer_skeleton is not None:
            sk = draw_skeleton_black(kps, W, H)
            writer_skeleton.write(sk)

        frame_idx += 1
        grabbed += 1
        if grabbed % 100 == 0:
            print(f"[info] processed {grabbed} sampled frames...")

    # 마무리
    if writer_preview is not None:
        writer_preview.release()
        print(f"[ok] saved preview: {PREVIEW_PATH}")
    if writer_skeleton is not None:
        writer_skeleton.release()
        print(f"[ok] saved skeleton: {SKELETON_PATH}")

    cap.release()

    # JSON 저장
    out_fps = fps
    out = {
        "source": os.path.basename(VIDEO_PATH),
        "video_size": [W, H],
        "fps": out_fps,              # fps 보정해서 저장
        "stride": STRIDE,
        "normalized": False,         # 좌표는 픽셀 단위
        "frames": frames_out
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)

    print(f"[ok] wrote {OUT_JSON}  (frames: {len(frames_out)}, fps={out_fps})")

if __name__ == "__main__":
    main()
