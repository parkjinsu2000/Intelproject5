import numpy as np
import math
import cv2
from numpy.linalg import norm

# 설정값
KPT_CONF_THRES = 0.20
K_STRICT = 8.0
MARGIN = 0.02

# COCO 키포인트 인덱스
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 5, 6, 7, 8, 9, 10
L_HP, R_HP, L_KN, R_KN, L_AN, R_AN = 11, 12, 13, 14, 15, 16
L_EYE, R_EYE, L_EAR, R_EAR = 1, 2, 3, 4

# 각도 계산용 트리플
ANGLE_TRIPLES = [
    (L_SH, L_EL, L_WR), (R_SH, R_EL, R_WR),
    (L_HP, L_KN, L_AN), (R_HP, R_KN, R_AN),
    (L_SH, L_HP, L_KN), (R_SH, R_HP, R_KN),
]

# 스켈레톤 연결선
EDGES = [
    (5,7),(7,9), (6,8),(8,10),
    (11,13),(13,15), (12,14),(14,16),
    (5,6), (11,12), (5,11),(6,12),
    (0,5),(0,6), (0,1),(0,2), (1,3),(2,4)
]

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
    angs = [angle_of(pts[i], pts[j], pts[k]) for (i,j,k) in ANGLE_TRIPLES]
    v = np.array(angs, dtype=np.float32)
    if np.any(~np.isfinite(v)):
        m = np.nanmean(v[np.isfinite(v)]) if np.any(np.isfinite(v)) else 0.0
        v = np.nan_to_num(v, nan=m)
    return v

def normalize_keypoints(pts):
    pts = pts.copy()
    if np.any(~np.isfinite(pts[[L_HP, R_HP]])):
        center = np.nanmean(pts[[L_SH, R_SH]], axis=0)
    else:
        center = np.nanmean(pts[[L_HP, R_HP]], axis=0)
    out = pts - center
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

def draw_pose_with_id(img, kps_xy, kps_conf, tid:int, box_xyxy=None, conf_thres=KPT_CONF_THRES):
    H, W = img.shape[:2]
    kpt_radius = max(2, int(min(H, W) * 0.004))
    line_thickness = max(2, int(min(H, W) * 0.003))
    # 스켈레톤
    for i, j in EDGES:
        if i < 17 and j < 17:
            pi, pj = kps_xy[i], kps_xy[j]
            if np.all(np.isfinite(pi)) and np.all(np.isfinite(pj)):
                cv2.line(img, tuple(np.round(pi).astype(int)),
                         tuple(np.round(pj).astype(int)),
                         (0, 255, 0), line_thickness, cv2.LINE_AA)
    # 키포인트
    for idx, p in enumerate(kps_xy):
        if not np.all(np.isfinite(p)):
            continue
        color = (0,255,0)
        if kps_conf is not None:
            c = kps_conf[idx]
            if (not np.isfinite(c)) or (c < conf_thres):
                color = (0,0,255)
        cv2.circle(img, tuple(np.round(p).astype(int)), kpt_radius, color, -1, cv2.LINE_AA)

    # 라벨 위치: 박스(top-left) 또는 키포인트 최소 좌표 근처
    if box_xyxy is not None:
        x1, y1, x2, y2 = box_xyxy
        lx, ly = int(x1), int(max(10, y1 - 8))
    else:
        finite = kps_xy[np.isfinite(kps_xy).all(axis=1)]
        if len(finite):
            x_min = int(np.min(finite[:,0]))
            y_min = int(np.min(finite[:,1]))
            lx, ly = x_min, max(10, y_min - 8)
        else:
            lx, ly = 12, 28

    label = f"ID {tid}"
    # 가독성 테두리
    cv2.putText(img, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(img, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

def put_text(img, text, org=(12,48), scale=1.0, color=(255,255,255)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)
