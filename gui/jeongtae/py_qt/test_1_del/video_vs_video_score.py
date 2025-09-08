#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
video_vs_video_score.py
- 정답 영상(ref.mp4)과 비교 영상(cmp.mp4)을 같은 시점 프레임끼리 바로 비교
- 매 N프레임(기본 5)마다 프레임 점수 s를 계산하고, 규칙에 따라 총점(초기 100)을 가감
- 화면 출력 없이 콘솔에 총점만 연속 출력
- 종료 시 최종 총점 출력

채점 규칙 (총점은 [0,100] 클램프):
  s < 50  → 총점 -1
  s >= 70 → 총점 +1
  50 ≤ s < 70 → 변화 없음
"""

import argparse, math
import numpy as np
import cv2
import torch
from ultralytics import YOLO
from numpy.linalg import norm

# -------------------- 설정 --------------------
MODEL_PATH_DEFAULT = "yolov8n-pose.pt"
DETECT_CONF_THRES = 0.25
KPT_CONF_THRES = 0.20

# 점수 매핑(빡빡)
K_STRICT = 8.0
MARGIN = 0.02

# COCO 17 keypoints index
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

def cosine_dist(a, b):
    a = np.nan_to_num(a); b = np.nan_to_num(b)
    return 1.0 - float(np.dot(a,b) / (norm(a)*norm(b) + 1e-6))

def frame_score_strict(vec_ref, vec_live, k=K_STRICT, margin=MARGIN):
    d_cos = cosine_dist(vec_ref, vec_live)
    ang_deg = float(np.degrees(np.mean(np.abs(vec_ref - vec_live))))
    pair_cost = 0.5*d_cos + 0.5*(ang_deg/180.0)
    d_eff = max(0.0, pair_cost - margin)
    score = 100.0 * math.exp(-k * d_eff)
    return float(np.clip(score, 0.0, 100.0))

def infer_one_pose(model, frame, imgsz, device, half):
    with torch.inference_mode():
        res = model.predict(frame, imgsz=imgsz, device=device, half=half,
                            conf=DETECT_CONF_THRES, verbose=False)[0]
    if (res.keypoints is None) or (len(res.keypoints) == 0):
        return None
    # 가장 큰 사람 한 명
    if len(res.boxes) > 1:
        areas = (res.boxes.xywh[:,2] * res.boxes.xywh[:,3]).detach().cpu().numpy()
        idx = int(np.argmax(areas))
    else:
        idx = 0
    kps = res.keypoints.xy[idx].detach().cpu().numpy()      # (17,2)
    conf = res.keypoints.conf[idx].detach().cpu().numpy()   # (17,)
    kps[conf < KPT_CONF_THRES] = np.nan
    kps_n = normalize_keypoints(kps)
    return pose_to_anglevec(kps_n)

# ------------------------------ 메인 ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref", help="정답 영상 경로 (예: ref.mp4)")
    ap.add_argument("cmp", help="비교 영상 경로 (예: compare.mp4)")
    ap.add_argument("--every", type=int, default=5, help="N프레임마다 채점 (기본 5)")
    ap.add_argument("--model", type=str, default=MODEL_PATH_DEFAULT, help="YOLO pose 가중치")
    ap.add_argument("--imgsz", type=int, default=320, help="YOLO 입력 해상도(작을수록 빠름)")
    ap.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--half", action="store_true", default=False, help="CUDA에서 FP16 사용")
    args = ap.parse_args()

    # 모델
    model = YOLO(args.model)
    try:
        model.to(args.device)  # type: ignore[attr-defined]
    except Exception:
        pass
    use_half = bool(args.half and args.device == "cuda")

    # 비디오 열기
    capR = cv2.VideoCapture(args.ref)
    capC = cv2.VideoCapture(args.cmp)
    if not capR.isOpened(): raise RuntimeError(f"Cannot open reference: {args.ref}")
    if not capC.isOpened(): raise RuntimeError(f"Cannot open compare: {args.cmp}")

    # 총점 초기화
    total_points = 100
    frame_idx = 0

    print("[info] start scoring (video vs. video)")
    while True:
        okR, fR = capR.read()
        okC, fC = capC.read()
        if not okR or not okC:
            # 어느 한쪽이 끝나면 종료
            if not okR: print("[info] reference ended.")
            if not okC: print("[info] compare ended.")
            break

        frame_idx += 1
        if frame_idx % max(1, args.every) != 0:
            continue  # 스킵

        # 두 프레임 포즈 → 각도 벡터
        vecR = infer_one_pose(model, fR, args.imgsz, args.device, use_half)
        vecC = infer_one_pose(model, fC, args.imgsz, args.device, use_half)

        if (vecR is None) or (vecC is None):
            # 한쪽이라도 포즈 없음 → 이번 틱은 점수 변화 없음
            print(f"[total] frame{frame_idx}: {total_points}  (no pose)")
            continue

        # 프레임 점수 산출
        s = frame_score_strict(vecR, vecC)

        # 총점 갱신 (클램프 포함)
        if s < 50.0:
            total_points = max(0, total_points - 1)
        elif s >= 70.0:
            total_points = min(100, total_points + 1)

        # 콘솔 출력(총점만)
        print(f"[total] frame{frame_idx}: {total_points}")

    capR.release(); capC.release()
    print(f"[final] total score = {total_points}")

if __name__ == "__main__":
    main()
