# -*- coding: utf-8 -*-
import time, math
import numpy as np
import cv2
from numpy.linalg import norm

from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage

# -------- 포즈/스코어 설정 --------
DETECT_CONF_THRES = 0.25
KPT_CONF_THRES = 0.20
K_STRICT = 8.0
MARGIN = 0.02

# 스켈레톤 연결 (COCO 17)
EDGES = [
    (5,7),(7,9), (6,8),(8,10),
    (11,13),(13,15), (12,14),(14,16),
    (5,6), (11,12), (5,11),(6,12),
    (0,5),(0,6), (0,1),(0,2), (1,3),(2,4)
]
NOSE=0; L_EYE=1; R_EYE=2; L_EAR=3; R_EAR=4
L_SH=5; R_SH=6; L_EL=7; R_EL=8; L_WR=9; R_WR=10
L_HP=11; R_HP=12; L_KN=13; R_KN=14; L_AN=15; R_AN=16

ANGLE_TRIPLES = [
    (L_SH, L_EL, L_WR),
    (R_SH, R_EL, R_WR),
    (L_HP, L_KN, L_AN),
    (R_HP, R_KN, R_AN),
    (L_SH, L_HP, L_KN),
    (R_SH, R_HP, R_KN),
]

def angle_of(a, b, c):
    if (a is None) or (b is None) or (c is None): return np.nan
    if np.any(~np.isfinite(a)) or np.any(~np.isfinite(b)) or np.any(~np.isfinite(c)): return np.nan
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

def draw_pose(img, kps_xy, kps_conf=None, conf_thres=KPT_CONF_THRES):
    H, W = img.shape[:2]
    kpt_radius = max(2, int(min(H, W) * 0.004))
    line_thickness = max(2, int(min(H, W) * 0.003))
    for i, j in EDGES:
        if i < 17 and j < 17:
            pi, pj = kps_xy[i], kps_xy[j]
            if np.all(np.isfinite(pi)) and np.all(np.isfinite(pj)):
                cv2.line(img, tuple(np.round(pi).astype(int)),
                         tuple(np.round(pj).astype(int)), (0, 255, 0),
                         line_thickness, cv2.LINE_AA)
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

# -------- YOLO --------
from ultralytics import YOLO

def get_main_pose_from_frame(frame, model):
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
    kps[conf < KPT_CONF_THRES] = np.nan
    return kps, conf

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
        times.append(ts); kps_list.append(kps); conf_list.append(conf)
    cap.release()
    angles = _interpolate_missing_vectors(angles)
    return {
        "times": np.array(times, dtype=np.float32),
        "angles": angles,
        "kps": kps_list, "conf": conf_list,
    }

class PoseWorker(QThread):
    # frame_ready = pyqtSignal(QImage) # 합성 프레임
    frame_ready = pyqtSignal(object)
    info_ready  = pyqtSignal(str)    # 상태/로그

    def __init__(self, model_path, ref_path, cam_index=0, mirror=True, start_sec=3.0, parent=None):
        super().__init__(parent)
        self.model_path = model_path
        self.ref_path = ref_path
        self.cam_index = cam_index
        self.mirror = mirror
        self.start_sec = start_sec
        self._running = True

    def stop(self):
        self._running = False

    def run(self):
        try:
            self.info_ready.emit("[info] loading model...")
            model = YOLO(self.model_path)
        except Exception as e:
            self.info_ready.emit(f"[error] load model: {e}")
            return

        self.info_ready.emit("[info] opening videos...")
        cap_ref = cv2.VideoCapture(self.ref_path)
        cap_cam = cv2.VideoCapture(self.cam_index)
        if not cap_ref.isOpened():
            self.info_ready.emit(f"[error] cannot open ref: {self.ref_path}")
            return
        if not cap_cam.isOpened():
            self.info_ready.emit("[error] cannot open camera")
            return

        fpsR = cap_ref.get(cv2.CAP_PROP_FPS) or 30.0
        W = int(max(cap_ref.get(cv2.CAP_PROP_FRAME_WIDTH),  640))
        H = int(max(cap_ref.get(cv2.CAP_PROP_FRAME_HEIGHT), 360))
        size_single = (W, H)

        start_t = time.monotonic()
        ema_score, ema_alpha = None, 0.2
        total_score, score_count = 0.0, 0

        self.info_ready.emit("[info] running...")
        while self._running:
            now = time.monotonic()
            elapsed = now - start_t
            t_ref = elapsed + self.start_sec

            # ---- ref 프레임: 해당 시각으로 seek 후 한 프레임만 처리
            cap_ref.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t_ref*1000.0))
            okR, fR = cap_ref.read()
            if not okR:
                self.info_ready.emit("[info] reference video ended.")
                break
            fR = cv2.resize(fR, size_single)
            kpsR, confR = get_main_pose_from_frame(fR, model)
            if kpsR is not None:
                draw_pose(fR, kpsR, kps_conf=confR)
            put_text(fR, "REF")

            # ---- live 프레임
            okC, fC = cap_cam.read()
            if not okC:
                self.info_ready.emit("[warn] camera read failed")
                break
            fC = cv2.resize(fC, size_single)
            kpsL, confL = get_main_pose_from_frame(fC, model)
            score_str = "--"

            if kpsL is not None:
                draw_pose(fC, kpsL, kps_conf=confL)

                if elapsed >= self.start_sec and kpsR is not None:
                    kps_use = flip_horizontal_pts(kpsL) if self.mirror else kpsL
                    vecL = pose_to_anglevec(normalize_keypoints(kps_use))
                    vecR = pose_to_anglevec(normalize_keypoints(kpsR))
                    s, _, _ = frame_score_strict(vecR, vecL)
                    ema_score = s if (ema_score is None) else (ema_score*(1-ema_alpha) + s*ema_alpha)
                    score_str = f"{ema_score:05.1f}"
                    total_score += s; score_count += 1

                mode_tag = "mirror" if self.mirror else "normal"
                put_text(fC, f"USER [{mode_tag}]   SCORE: {score_str}")
            else:
                put_text(fC, "USER [no person]")

            # ---- 합성 → QImage emit
            canvas = np.hstack([fR, fC])
            rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888).copy()
            self.frame_ready.emit(qimg)

            # 속도 조절
            self.msleep(int(max(1, 1000.0/float(min(fpsR, 30)))) )

        cap_ref.release(); cap_cam.release()

        if score_count > 0:
            final_avg = total_score / score_count
            self.info_ready.emit(f"[final] average score = {final_avg:.2f} over {score_count} frames")
        else:
            self.info_ready.emit("[final] no scored frames")

