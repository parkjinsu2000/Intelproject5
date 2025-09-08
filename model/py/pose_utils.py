
import numpy as np
from numpy.linalg import norm
from settings import ANGLE_TRIPLES, K_STRICT, MARGIN, L_HP, R_HP, L_SH, R_SH

def angle_of(a,b,c):
    if a is None or b is None or c is None: return np.nan
    if np.any(~np.isfinite(a)) or np.any(~np.isfinite(b)) or np.any(~np.isfinite(c)):
        return np.nan
    v1 = a-b; v2 = c-b
    n1 = norm(v1); n2 = norm(v2)
    if n1<1e-6 or n2<1e-6: return np.nan
    cosv = np.clip(np.dot(v1,v2)/(n1*n2),-1.0,1.0)
    return np.arccos(cosv)

def pose_to_anglevec(pts):
    angs=[]
    for i,j,k in ANGLE_TRIPLES: angs.append(angle_of(pts[i],pts[j],pts[k]))
    v = np.array(angs,dtype=np.float32)
    if np.any(~np.isfinite(v)):
        m = np.nanmean(v[np.isfinite(v)]) if np.any(np.isfinite(v)) else 0.0
        v = np.nan_to_num(v,nan=m)
    return v

def normalize_keypoints(pts):
    pts = pts.copy()
    if np.any(~np.isfinite(pts[[L_HP,R_HP]])):
        center = np.nanmean(pts[[L_SH,R_SH]],axis=0)
    else:
        center = np.nanmean(pts[[L_HP,R_HP]],axis=0)
    out = pts - center
    if np.any(~np.isfinite(pts[[L_SH,R_SH]])):
        finite = pts[np.isfinite(pts).all(axis=1)]
        scale = np.max(norm(finite-finite.mean(0),axis=1)) if len(finite) else 1.0
    else:
        scale = norm(pts[L_SH]-pts[R_SH])
        if not np.isfinite(scale) or scale<1e-6:
            finite = pts[np.isfinite(pts).all(axis=1)]
            scale = np.max(norm(finite-finite.mean(0),axis=1)) if len(finite) else 1.0
    return out/(scale+1e-6)

def cosine_dist(a,b):
    a=np.nan_to_num(a); b=np.nan_to_num(b)
    return 1.0 - float(np.dot(a,b)/(norm(a)*norm(b)+1e-6))

def frame_score_strict(vec_ref, vec_live,k=K_STRICT,margin=MARGIN):
    d_cos = cosine_dist(vec_ref,vec_live)
    ang_deg = float(np.degrees(np.mean(np.abs(vec_ref-vec_live))))
    pair_cost = 0.5*d_cos + 0.5*(ang_deg/180.0)
    d_eff = max(0.0,pair_cost-margin)
    score = 100.0* np.exp(-k*d_eff)
    return float(np.clip(score,0.0,100.0)), pair_cost, ang_deg
