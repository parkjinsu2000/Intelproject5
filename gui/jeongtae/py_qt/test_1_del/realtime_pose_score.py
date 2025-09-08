#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
realtime_pose_score_sync.py
- 정답 영상(ref.mp4)과 실시간 카메라 영상을 동시에 재생
- 매 N 프레임(기본 5)마다 두 프레임의 포즈 similarity를 계산해 점수 산출
- 카운트다운 동안 REF 정지 → START 후 재생
- ★채점 규칙:
    - 총점은 100에서 시작
    - s < 50  → 총점 -1
    - s >= 70 → 총점 +1
    - 50 ≤ s < 70 → 변화 없음
    - 총점 범위는 [0, 100]
- 화면엔 점수/총점 표시 없음, 종료 시 총점 출력
- 콘솔에 실시간 총점 print
"""

import argparse, time, math
import numpy as np
import cv2
import base64       # base64 인코딩
import sys          # 시스템 입출력 제어
import torch
from ultralytics import YOLO
from numpy.linalg import norm

from PyQt5.QtWidgets import QLabel, QApplication, QWidget, QVBoxLayout
from PyQt5.QtGui import QImage, QPixmap


# -------------------- 설정 --------------------
MODEL_PATH = "yolov8l-pose.pt"
DETECT_CONF_THRES = 0.25
KPT_CONF_THRES = 0.20

# 점수 매핑(빡빡)
K_STRICT = 8.0
MARGIN = 0.02

# COCO 17 keypoints
NOSE=0; L_EYE=1; R_EYE=2; L_EAR=3; R_EAR=4
L_SH=5; R_SH=6; L_EL=7; R_EL=8
L_WR=9; R_WR=10; L_HP=11; R_HP=12
L_KN=13; R_KN=14; L_AN=15; R_AN=16

# 각도 계산용 (∠i-j-k at j)
ANGLE_TRIPLES = [
    (L_SH, L_EL, L_WR), (R_SH, R_EL, R_WR),
    (L_HP, L_KN, L_AN), (R_HP, R_KN, R_AN),
    (L_SH, L_HP, L_KN), (R_SH, R_HP, R_KN),
]

# 시각화용 엣지
EDGES = [
    (5,7),(7,9), (6,8),(8,10),
    (11,13),(13,15), (12,14),(14,16),
    (5,6), (11,12), (5,11),(6,12),
    (0,5),(0,6), (0,1),(0,2), (1,3),(2,4)
]

# -------------------- 유틸 --------------------
def angle_of(a, b, c):
    if (a is None) or (b is None) or (c is None): return np.nan
    if np.any(~np.isfinite(a)) or np.any(~np.isfinite(b)) or np.any(~np.isfinite(c)):
        return np.nan
    v1 = a - b; v2 = c - b
    n1 = norm(v1); n2 = norm(v2)
    if n1 < 1e-6 or n2 < 1e-6: return np.nan
    cosv = np.clip(np.dot(v1, v2) / (n1*n2), -1.0, 1.0)
    return math.acos(cosv)

def pose_to_anglevec(pts):
    angs = []
    for (i,j,k) in ANGLE_TRIPLES:
        angs.append(angle_of(pts[i], pts[j], pts[k]))
    v = np.array(angs, dtype=np.float32)
    if np.any(~np.isfinite(v)):
        m = np.nanmean(v[np.isfinite(v)]) if np.any(np.isfinite(v)) else 0.0
        v = np.nan_to_num(v, nan=m)
    return v

def normalize_keypoints(pts):
    pts = pts.copy()
    # 중심: 골반(없으면 어깨)
    if np.any(~np.isfinite(pts[[L_HP, R_HP]])):
        center = np.nanmean(pts[[L_SH, R_SH]], axis=0)
    else:
        center = np.nanmean(pts[[L_HP, R_HP]], axis=0)
    out = pts - center
    # 스케일: 어깨폭(없으면 분산 반경)
    if np.any(~np.isfinite(pts[[L_SH, R_SH]])):
        finite = pts[np.isfinite(pts).all(axis=1)]
        scale = np.max(norm(finite - finite.mean(0), axis=1)) if len(finite) else 1.0
    else:
        scale = norm(pts[L_SH] - pts[R_SH])
        if not np.isfinite(scale) or scale < 1e-6:
            finite = pts[np.isfinite(pts).all(axis=1)]
            scale = np.max(norm(finite - finite.mean(0), axis=1)) if len(finite) else 1.0
    return out / (scale + 1e-6)

def flip_horizontal_pts(pts):
    out = pts.copy()
    out[:,0] *= -1
    for l,r in [(L_SH,R_SH),(L_EL,R_EL),(L_WR,R_WR),
                (L_HP,R_HP),(L_KN,R_KN),(L_AN,R_AN),
                (L_EYE,R_EYE),(L_EAR,R_EAR)]:
        out[[l,r]] = out[[r,l]]
    return out

def cosine_dist(a, b):
    a = np.nan_to_num(a); b = np.nan_to_num(b)
    return 1.0 - float(np.dot(a,b) / (norm(a)*norm(b) + 1e-6))

def frame_score_strict(vec_ref, vec_live, k=K_STRICT, margin=MARGIN):
    d_cos = cosine_dist(vec_ref, vec_live)
    ang_deg = float(np.degrees(np.mean(np.abs(vec_ref - vec_live))))
    pair_cost = 0.5*d_cos + 0.5*(ang_deg/180.0)
    d_eff = max(0.0, pair_cost - margin)
    score = 100.0 * math.exp(-k * d_eff)
    return float(np.clip(score, 0.0, 100.0)), pair_cost, ang_deg

def draw_pose(img, kps_xy, kps_conf=None, conf_thres=KPT_CONF_THRES):
    H, W = img.shape[:2]
    kpt_radius = max(2, int(min(H, W) * 0.004))
    line_thickness = max(2, int(min(H, W) * 0.003))
    for i, j in EDGES:
        if i < 17 and j < 17:
            pi, pj = kps_xy[i], kps_xy[j]
            if np.all(np.isfinite(pi)) and np.all(np.isfinite(pj)):
                cv2.line(img, tuple(np.round(pi).astype(int)),
                         tuple(np.round(pj).astype(int)),
                         (0, 255, 0), line_thickness, cv2.LINE_AA)
    for idx, p in enumerate(kps_xy):
        if not np.all(np.isfinite(p)):
            continue
        color = (0,255,0)
        if kps_conf is not None:
            c = kps_conf[idx]
            if (not np.isfinite(c)) or (c < conf_thres):
                color = (0,0,255)
        cv2.circle(img, tuple(np.round(p).astype(int)), kpt_radius, color, -1, cv2.LINE_AA)
    return img

def put_text(img, text, org=(12,48), scale=1.0, color=(255,255,255)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)

# ---------------- YOLO 추론 공통 함수 ----------------
def make_infer(model, args, use_half: bool):
    def infer_pose(frame):
        with torch.inference_mode():
            res = model.predict(
                frame, imgsz=args.imgsz, device=args.device,
                half=use_half, conf=DETECT_CONF_THRES, verbose=False
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



    if writer is not None:
        writer.release()
    capR.release(); capU.release()
    cv2.destroyAllWindows()

    # 최종 총점 출력
    # print(f"[final] total score = {total_points}")

if __name__ == "__main__":
    main()
