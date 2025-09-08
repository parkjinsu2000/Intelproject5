#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
realtime_pose_score.py
- 정답 영상(ref.mp4)과 실시간 카메라 영상을 동시에 표시
- 3초 이후부터 실시간 점수 계산 (옵션 --start)
- 거울 모드(미러) 옵션: 카메라 포즈를 좌우 반전하여 채점 (표시는 그대로)
- (옵션) 자동 타이밍 보정: 최근 윈도우의 '움직임 에너지' 교차상관으로 오프셋 추정
- 합성 영상 저장 옵션
- 실행 종료 시 전체 프레임의 '평균 점수'를 최종 점수로 출력
"""

import argparse, time, math, collections
import numpy as np
import cv2
from ultralytics import YOLO
from numpy.linalg import norm

# ---------------------- 설정 ----------------------
MODEL_PATH = "yolov8l-pose.pt"
DETECT_CONF_THRES = 0.25
KPT_CONF_THRES = 0.20

# 엄격한 실시간 프레임 점수 맵핑(빡빡)
K_STRICT = 8.0
MARGIN = 0.02

# 스켈레톤 연결(ultralytics COCO 17)
EDGES = [
    (5,7),(7,9), (6,8),(8,10),
    (11,13),(13,15), (12,14),(14,16),
    (5,6), (11,12), (5,11),(6,12),
    (0,5),(0,6), (0,1),(0,2), (1,3),(2,4)
]

# COCO keypoint indices
NOSE=0; L_EYE=1; R_EYE=2; L_EAR=3; R_EAR=4
L_SH=5; R_SH=6; L_EL=7; R_EL=8
L_WR=9; R_WR=10; L_HP=11; R_HP=12
L_KN=13; R_KN=14; L_AN=15; R_AN=16

# 각도 계산용 삼각형(∠i-j-k at j)
ANGLE_TRIPLES = [
    (L_SH, L_EL, L_WR),   # 왼팔
    (R_SH, R_EL, R_WR),   # 오른팔
    (L_HP, L_KN, L_AN),   # 왼다리
    (R_HP, R_KN, R_AN),   # 오른다리
    (L_SH, L_HP, L_KN),   # 몸통-왼다리
    (R_SH, R_HP, R_KN),   # 몸통-오른다리
]

# --------------------------------------------------

def get_main_pose_from_frame(frame, model):
    """프레임에서 가장 큰 사람 1명의 (kps_xy, kps_conf) 반환. 없으면 (None,None)."""
    res = model(frame, conf=DETECT_CONF_THRES, verbose=False)[0]
    if (res.keypoints is None) or (len(res.keypoints) == 0):
        return None, None
    if len(res.boxes) > 1:
        areas = (res.boxes.xywh[:,2] * res.boxes.xywh[:,3]).cpu().numpy()
        idx = int(np.argmax(areas))
    else:
        idx = 0
    kps = res.keypoints.xy[idx].cpu().numpy()      # (17,2)
    conf = res.keypoints.conf[idx].cpu().numpy()   # (17,)
    # 저신뢰 → NaN 좌표로(각도 제외)
    kps[conf < KPT_CONF_THRES] = np.nan
    return kps, conf

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

def cosine_dist(a, b):
    a = np.nan_to_num(a); b = np.nan_to_num(b)
    return 1.0 - float(np.dot(a,b) / (norm(a)*norm(b) + 1e-6))

def frame_score_strict(vec_ref, vec_live, k=K_STRICT, margin=MARGIN):
    # 코사인 + 각도오차 혼합 코스트
    d_cos = cosine_dist(vec_ref, vec_live)
    ang_deg = float(np.degrees(np.mean(np.abs(vec_ref - vec_live))))
    pair_cost = 0.5*d_cos + 0.5*(ang_deg/180.0)
    d_eff = max(0.0, pair_cost - margin)
    score = 100.0 * math.exp(-k * d_eff)
    return float(np.clip(score, 0.0, 100.0)), pair_cost, ang_deg

def motion_energy_from_angles(vec_now, vec_prev):
    if vec_now is None or vec_prev is None: return 0.0
    diff = np.nan_to_num(vec_now) - np.nan_to_num(vec_prev)
    return float(np.linalg.norm(diff))

def draw_pose(img, kps_xy, kps_conf=None, conf_thres=KPT_CONF_THRES):
    H, W = img.shape[:2]
    kpt_radius = max(2, int(min(H, W) * 0.004))
    line_thickness = max(2, int(min(H, W) * 0.003))
    # 선
    for i, j in EDGES:
        if i < 17 and j < 17:
            pi, pj = kps_xy[i], kps_xy[j]
            if np.all(np.isfinite(pi)) and np.all(np.isfinite(pj)):
                cv2.line(img, tuple(np.round(pi).astype(int)),
                         tuple(np.round(pj).astype(int)),
                         (0, 255, 0), line_thickness, cv2.LINE_AA)
    # 점
    for idx, p in enumerate(kps_xy):
        if not np.all(np.isfinite(p)):  # missing
            continue
        color = (0,255,0)
        if kps_conf is not None:
            c = kps_conf[idx]
            if (not np.isfinite(c)) or (c < conf_thres):
                color = (0,0,255)
        cv2.circle(img, tuple(np.round(p).astype(int)), kpt_radius, color, -1, cv2.LINE_AA)
    return img

def put_text(img, text, org=(12,48), scale=1.1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255,255,255), 2, cv2.LINE_AA)

# --------- ref 전처리: 모든 프레임 포즈/각도/타임스탬프 ---------
def preprocess_reference(ref_path, model):
    cap = cv2.VideoCapture(ref_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open reference: {ref_path}")
    times, angles, kps_list, conf_list = [], [], [], []
    while True:
        ok, frame = cap.read()
        if not ok: break
        ts = cap.get(cv2.CAP_PROP_POS_MSEC)/1000.0
        kps, conf = get_main_pose_from_frame(frame, model)
        if kps is None:
            times.append(ts); angles.append(None); kps_list.append(None); conf_list.append(None)
            continue
        kps_n = normalize_keypoints(kps)
        angles.append(pose_to_anglevec(kps_n))
        times.append(ts)
        kps_list.append(kps); conf_list.append(conf)
    cap.release()
    # 각도 결측 보간
    angles = _interpolate_missing_vectors(angles)
    return {
        "times": np.array(times, dtype=np.float32),
        "angles": angles,                     # (T,D)
        "kps": kps_list, "conf": conf_list,   # 길이 T의 리스트(원본 좌표)
    }

def _interpolate_missing_vectors(seq):
    if len(seq) == 0: return np.zeros((0,1), dtype=np.float32)
    D = None
    for v in seq:
        if isinstance(v, np.ndarray):
            D = v.shape[0]; break
    if D is None: return np.zeros((len(seq),1), dtype=np.float32)
    X = []
    for v in seq:
        if v is None: X.append(np.full(D, np.nan, dtype=np.float32))
        else: X.append(v.astype(np.float32))
    X = np.vstack(X)
    for d in range(D):
        col = X[:, d]
        idx = np.where(np.isfinite(col))[0]
        if len(idx)==0: X[:, d] = 0.0
        else: X[:, d] = np.interp(np.arange(len(col)), idx, col[idx])
    return X

# --------- 슬라이딩 교차상관으로 오프셋 추정(실시간) ---------
def estimate_offset_sliding(live_energy_buf, live_dt, ref_times, ref_energy,
                            t_ref_now, max_lag=1.0):
    """
    live_energy_buf: 최근 W개 에너지 배열(등간격), live_dt: 샘플 간격(초)
    ref_energy는 ref_times에 맞춰 등간격이 아닐 수 있어 선형보간으로 구간 추출
    t_ref_now 주변에서 ±max_lag 범위로 best offset 찾기(양수=live가 늦음)
    """
    if len(live_energy_buf) < 5: return 0.0
    W = len(live_energy_buf) * live_dt
    # ref에서 [t_ref_now - W - max_lag, t_ref_now + max_lag] 구간 신호를 등간격으로 샘플
    t0 = max(ref_times[0], t_ref_now - W - max_lag)
    t1 = min(ref_times[-1], t_ref_now + max_lag)
    if t1 - t0 < W/2:  # 너무 짧으면 패스
        return 0.0
    # 등간격 grid로 보간
    grid = np.arange(t0, t1, live_dt, dtype=np.float32)
    ref_e = np.interp(grid, ref_times, ref_energy)
    live_e = np.array(live_energy_buf, dtype=np.float32)
    # 길이 맞추기
    n = min(len(ref_e), len(live_e))
    if n < 5: return 0.0
    ref_e = ref_e[:n]; live_e = live_e[:n]
    # 표준화
    def z(x):
        s = np.std(x)
        return (x - np.mean(x)) / (s + 1e-6) if s > 1e-6 else x*0
    ref_z = z(ref_e); live_z = z(live_e)
    # full xcorr
    corr = np.correlate(ref_z, live_z, mode='full')
    lags = np.arange(-n+1, n) * live_dt
    # lags>0 : live를 앞으로 당겨야함(=live가 늦음)
    best_lag = lags[int(np.argmax(corr))]
    # 제한
    best_lag = float(np.clip(best_lag, -max_lag, max_lag))
    return best_lag

# ------------------------------ 메인 ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref", help="정답 영상 경로 (예: ref.mp4)")
    ap.add_argument("--cam", type=int, default=0, help="카메라 인덱스 (기본 0)")
    ap.add_argument("--start", type=float, default=3.0, help="채점 시작 시각(초)")
    ap.add_argument("--no-mirror", action="store_true", help="거울 모드 비활성화(기본: 활성화)")
    ap.add_argument("--autosync", action="store_true", help="실시간 자동 타이밍 보정 사용")
    ap.add_argument("--sync-win", type=float, default=2.0, help="자동보정 윈도우 길이(초)")
    ap.add_argument("--sync-every", type=float, default=0.5, help="보정 주기(초)")
    ap.add_argument("--max-lag", type=float, default=1.0, help="최대 지연 탐색(초)")
    ap.add_argument("--save", type=str, default=None, help="합성 영상 저장 경로 (예: out.mp4)")
    args = ap.parse_args()

    mirror = (not args.no_mirror)

    print("[info] loading model...")
    model = YOLO(MODEL_PATH)

    # 1) 참조 전처리
    print("[info] preprocessing reference (pose & angles)...")
    ref = preprocess_reference(args.ref, model)
    ref_times = ref["times"]
    ref_angles = ref["angles"]
    # ref 움직임 에너지(프레임 간 차분 norm)
    ref_energy = np.zeros(len(ref_times), dtype=np.float32)
    for i in range(1, len(ref_times)):
        ref_energy[i] = motion_energy_from_angles(ref_angles[i], ref_angles[i-1])

    # 2) 비디오 핸들러
    cap_ref = cv2.VideoCapture(args.ref)
    cap_cam = cv2.VideoCapture(args.cam)
    if not cap_cam.isOpened():
        raise RuntimeError("Cannot open camera")

    fpsR = cap_ref.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(max(cap_ref.get(cv2.CAP_PROP_FRAME_WIDTH),  640))
    H = int(max(cap_ref.get(cv2.CAP_PROP_FRAME_HEIGHT), 360))
    size_single = (W, H)

    # 저장 옵션
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, int(round(fpsR)), (W*2, H))

    # 3) 루프 상태
    start_t = time.monotonic()
    ema_score = None
    ema_alpha = 0.2  # 점수 표기 부드럽게

    # ⬇⬇ 평균 최종 점수를 위한 누적 변수
    total_score = 0.0
    score_count = 0

    # autosync 상태
    offset_live = 0.0 # live가 ref 대비 늦는(+)/빠른(-) 시간(초)
    last_sync_t = 0.0
    # 에너지 버퍼(최근 윈도우), 샘플 간격은 동적으로 계산
    energy_buf = collections.deque(maxlen=max(5, int(args.sync_win * 20)))  # 대략 20Hz 가정
    last_angle_live = None

    print("[info] press 'q' to quit.")
    while True:
        # 현재 시간 기준 ref 시각 계산
        now = time.monotonic()
        elapsed = now - start_t
        t_ref = elapsed + args.start  # ref 기준 시작 offset
        # autosync 적용
        t_ref_sync = t_ref + offset_live

        # ---- ref 프레임 얻기(랜덤접근) ----
        cap_ref.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t_ref_sync*1000.0))
        okR, fR = cap_ref.read()
        if not okR:
            # ref 끝났으면 종료
            print("[info] reference video ended.")
            break
        fR = cv2.resize(fR, size_single)

        # ref 시각에 가장 가까운 인덱스
        idx_ref = int(np.searchsorted(ref_times, t_ref_sync))
        idx_ref = int(np.clip(idx_ref, 0, len(ref_times)-1))
        kpsR = ref["kps"][idx_ref]
        confR = ref["conf"][idx_ref]
        if kpsR is not None:
            draw_pose(fR, kpsR, kps_conf=confR)
        put_text(fR, "REF")

        # ---- live 프레임 ----
        okC, fC = cap_cam.read()
        if not okC:
            print("[warn] camera read failed")
            break
        fC = cv2.resize(fC, size_single)
        kpsL, confL = get_main_pose_from_frame(fC, model)
        if kpsL is not None:
            # 채점은 시작 시각 이후만
            score_str = "--"
            if elapsed >= args.start:
                kps_use = flip_horizontal_pts(kpsL) if mirror else kpsL
                vecL = pose_to_anglevec(normalize_keypoints(kps_use))
                vecR = ref_angles[idx_ref]
                s, pair_cost, ang_deg = frame_score_strict(vecR, vecL)
                ema_score = s if (ema_score is None) else (ema_score*(1-ema_alpha) + s*ema_alpha)
                score_str = f"{ema_score:05.1f}"

                # ⬇⬇ 평균용 누적
                total_score += s
                score_count += 1

                # autosync: 에너지 업데이트 및 주기적 보정
                if args.autosync:
                    e_now = motion_energy_from_angles(vecL, last_angle_live)
                    energy_buf.append(e_now)
                    if (now - last_sync_t) >= args.sync_every and len(energy_buf) >= 5:
                        # live 샘플 간격 추정: 윈도우 길이 / 샘플 수
                        live_dt = max(1e-3, float(args.sync_win) / float(len(energy_buf)))
                        best_lag = estimate_offset_sliding(
                            list(energy_buf), live_dt, ref_times, ref_energy,
                            t_ref_sync, max_lag=args.max_lag
                        )
                        # 완전 치환 대신 EMA로 부드럽게 반영
                        offset_live = 0.7*offset_live + 0.3*best_lag
                        last_sync_t = now
                last_angle_live = vecL

            draw_pose(fC, kpsL, kps_conf=confL)
            mode_tag = "mirror" if mirror else "normal"
            put_text(fC, f"USER [{mode_tag}]   SCORE: {score_str}")
        else:
            put_text(fC, "USER [no person]")

        # ---- 합성 및 표시/저장 ----
        canvas = np.hstack([fR, fC])
        cv2.imshow("DanceCompare (press q to quit)", canvas)
        if writer is not None:
            writer.write(canvas)

        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    # 저장기 종료/자원 해제
    if writer is not None:
        writer.release()
    cap_ref.release()
    cap_cam.release()
    cv2.destroyAllWindows()

    # ⬇⬇ 최종 평균 점수 출력
    if score_count > 0:
        final_avg = total_score / score_count
        print(f"[final] average score = {final_avg:.2f} (over {score_count} scored frames)")
    else:
        print("[final] no scored frames (person not detected after start time)")

if __name__ == "__main__":
    main()
