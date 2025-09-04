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
- 실시간 총점 오버레이(상단 중앙), 종료 시 총점 출력
"""

import argparse, time, math
import numpy as np
import cv2
import torch
from ultralytics import YOLO
from numpy.linalg import norm

# -------------------- 설정 --------------------
MODEL_PATH = "yolov8n-pose.pt"
DETECT_CONF_THRES = 0.25
KPT_CONF_THRES = 0.20

# 점수 매핑(빡빡)
K_STRICT = 8.0
MARGIN = 0.02
ALERT_THRESH = 50.0   # 50점 미만이면 느낌표 표시

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

def draw_exclamation(img, color=(0, 0, 255)):
    H, W = img.shape[:2]
    text = "!"
    scale = max(1.0, min(H, W) / 220.0)
    thickness = max(2, int(scale * 4))
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = (W - tw) // 2
    y = (H + th) // 2
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

def draw_countdown_overlay(img, text):
    H, W = img.shape[:2]
    scale = max(1.0, min(H, W) / 180.0)
    thickness = max(2, int(scale * 4))
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = (W - tw) // 2
    y = (H + th) // 2
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thickness+2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255,255,255), thickness, cv2.LINE_AA)

# 총점 오버레이(가운데 위쪽)
def draw_total_overlay_center_top(img, total_points):
    H, W = img.shape[:2]
    text = f"TOTAL: {int(total_points)}"
    scale = max(1.0, min(H, W) / 700.0)
    thickness = max(2, int(scale * 4))
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = (W - tw) // 2
    y = max(th + 10, int(H * 0.08))
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thickness+2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255,255,255), thickness, cv2.LINE_AA)

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

# ------------------------------ 메인 ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref", help="정답 영상 경로 (예: ref.mp4)")
    ap.add_argument("--cam", type=int, default=0, help="카메라 인덱스 (기본 0)")
    ap.add_argument("--start", type=float, default=3.0, help="카운트다운 길이(초)")
    ap.add_argument("--every", type=int, default=5, help="N프레임마다 채점 (기본 5)")
    ap.add_argument("--no-mirror", action="store_true", help="거울 모드 비활성화(기본: 활성화)")
    # 표시/저장
    ap.add_argument("--disp-scale", type=float, default=1.0, help="표시 스케일(예: 0.7)")
    ap.add_argument("--disp-width", type=int, default=None, help="표시 가로 최대폭(px)")
    ap.add_argument("--save", type=str, default=None, help="합성 영상 저장 경로 (예: out.mp4)")
    # 성능
    ap.add_argument("--imgsz", type=int, default=320, help="YOLO 입력 해상도(작을수록 빠름)")
    ap.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--half", action="store_true", default=True, help="CUDA에서 FP16 사용")
    args = ap.parse_args()

    mirror = (not args.no_mirror)

    # OpenCV 최적화
    cv2.setUseOptimized(True); cv2.setNumThreads(0)

    # 모델 로드
    print("[info] loading model...")
    model = YOLO(MODEL_PATH)
    try:
        model.to(args.device)  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        model.fuse()
    except Exception:
        pass
    use_half = bool(args.half and args.device == "cuda")
    if use_half:
        try:
            model.model.half()
        except Exception:
            use_half = False

    # 간단 워머프
    _dummy = np.zeros((384, 384, 3), np.uint8)
    with torch.inference_mode():
        for _ in range(2):
            _ = model.predict(_dummy, imgsz=args.imgsz, device=args.device,
                              half=use_half, conf=DETECT_CONF_THRES, verbose=False)

    infer_pose = make_infer(model, args, use_half)

    # 캡처 열기
    capR = cv2.VideoCapture(args.ref)
    capU = cv2.VideoCapture(args.cam)
    if not capR.isOpened(): raise RuntimeError(f"Cannot open reference: {args.ref}")
    if not capU.isOpened(): raise RuntimeError("Cannot open camera")

    # 사이즈/FPS
    fpsR = capR.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(max(capR.get(cv2.CAP_PROP_FRAME_WIDTH),  640))
    H = int(max(capR.get(cv2.CAP_PROP_FRAME_HEIGHT), 360))
    size_single = (W, H)

    # 저장기
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, int(round(fpsR)), (W*2, H))

    print("[info] press 'q' to quit.")

    # ---------------- 카운트다운: REF 정지 프레임 표시 ----------------
    okR0, firstR = capR.read()
    if not okR0:
        raise RuntimeError("Reference video has no frames.")
    firstR = cv2.resize(firstR, size_single)

    # 총점 초기화
    total_points = 100

    t0 = time.monotonic()
    countdown = max(0.0, float(args.start))
    while True:
        now = time.monotonic()
        elapsed = now - t0
        remain = countdown - elapsed
        if remain <= -0.2:
            break

        okU, fU = capU.read()
        if not okU:
            continue
        fU = cv2.resize(fU, size_single)

        fL = firstR.copy()
        fR = fU.copy()

        text = str(int(np.ceil(remain))) if remain > 0.8 else "START"
        draw_countdown_overlay(fL, text)
        draw_countdown_overlay(fR, text)

        put_text(fL, "REF", (12,48), 1.1)
        put_text(fR, f"USER [{'mirror' if mirror else 'normal'}]", (12,48), 1.0)

        canvas = np.hstack([fL, fR])

        # 총점 오버레이
        draw_total_overlay_center_top(canvas, total_points)

        disp = canvas
        scale = args.disp_scale
        if args.disp_width is not None and args.disp_width > 0:
            scale = min(scale, float(args.disp_width) / float(canvas.shape[1]))
        if scale < 1.0:
            new_w = int(canvas.shape[1] * scale)
            new_h = int(canvas.shape[0] * scale)
            disp = cv2.resize(canvas, (new_w, new_h), interpolation=cv2.INTER_AREA)

        cv2.imshow("DanceCompare Sync (press q to quit)", disp)
        if writer is not None:
            writer.write(canvas)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            if writer is not None: writer.release()
            capR.release(); capU.release()
            cv2.destroyAllWindows()
            return

    # 카운트다운 끝 → REF를 0프레임부터 재생 시작
    capR.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # ---------------- 본 루프 ----------------
    start_t = time.monotonic()      # START 시각
    frame_idx = 0

    ema_score = None
    ema_alpha = 0.25
    lastR = {"kps": None, "conf": None}
    lastU = {"kps": None, "conf": None}
    alert_active = False

    while True:
        okR, fR = capR.read()
        okU, fU = capU.read()
        if not okR or not okU:
            if not okR: print("[info] reference video ended.")
            if not okU: print("[warn] camera read failed.")
            break

        fR = cv2.resize(fR, size_single)
        fU = cv2.resize(fU, size_single)

        elapsed = time.monotonic() - start_t
        scoring_on = (elapsed >= 0.0)  # START 이후 즉시 채점

        frame_idx += 1
        do_score = (frame_idx % max(1, args.every) == 0)

        if do_score:
            kpsR, confR = infer_pose(fR)
            kpsU, confU = infer_pose(fU)

            if kpsR is not None: lastR = {"kps": kpsR, "conf": confR}
            if kpsU is not None: lastU = {"kps": kpsU, "conf": confU}

            if scoring_on and (lastR["kps"] is not None) and (lastU["kps"] is not None):
                kR_n = normalize_keypoints(lastR["kps"])
                kU_use = flip_horizontal_pts(lastU["kps"]) if mirror else lastU["kps"]
                kU_n = normalize_keypoints(kU_use)

                vecR = pose_to_anglevec(kR_n)
                vecU = pose_to_anglevec(kU_n)

                s, pair_cost, ang_deg = frame_score_strict(vecR, vecU)

                # EMA는 경고 깜빡임 완화용
                ema_score = s if (ema_score is None) else (ema_score*(1-ema_alpha) + s*ema_alpha)

                # ★ 감점/가점 규칙
                if s < 50.0:
                    total_points = max(0, total_points - 1)
                elif s >= 70.0:
                    total_points = min(100, total_points + 1)

                # 경고(느낌표) 여부(EMA 기준)
                show_for_alert = ema_score if ema_score is not None else s
                alert_active = bool(show_for_alert < ALERT_THRESH)
            else:
                alert_active = False

        if lastR["kps"] is not None:
            draw_pose(fR, lastR["kps"], kps_conf=lastR["conf"])
        if lastU["kps"] is not None:
            draw_pose(fU, lastU["kps"], kps_conf=lastU["conf"])

        put_text(fR, "REF", (12,48), 1.1)
        put_text(fU, f"USER [{'mirror' if mirror else 'normal'}]", (12,48), 1.0)

        if scoring_on and alert_active:
            draw_exclamation(fU)

        canvas = np.hstack([fR, fU])

        # 총점 오버레이(항상)
        draw_total_overlay_center_top(canvas, total_points)

        disp = canvas
        scale = args.disp_scale
        if args.disp_width is not None and args.disp_width > 0:
            scale = min(scale, float(args.disp_width) / float(canvas.shape[1]))
        if scale < 1.0:
            new_w = int(canvas.shape[1] * scale)
            new_h = int(canvas.shape[0] * scale)
            disp = cv2.resize(canvas, (new_w, new_h), interpolation=cv2.INTER_AREA)

        cv2.imshow("DanceCompare Sync (press q to quit)", disp)
        if writer is not None:
            writer.write(canvas)

        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    if writer is not None:
        writer.release()
    capR.release(); capU.release()
    cv2.destroyAllWindows()

    # 최종: 총점만 출력
    print(f"[final] total score = {total_points}")

if __name__ == "__main__":
    main()
