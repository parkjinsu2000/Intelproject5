#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mannequin.py — JSON 길이 반영 + anchors.json 로드 + 배경 이미지(anchors.json에서 설정)
- 파츠/앵커/팁/기준해상도(ref_size)/배경(background)는 ASSETS_DIR/anchors.json에서 로드
- 파츠는 회전/이동/스케일까지 적용해 JSON 포즈에 맞춤
"""

import os, json
import numpy as np
import cv2

# ===== 기본 설정 (anchors.json에서 ref_size를 읽어 덮어씀) =====
JSON_PATH   = "naruto.json"
ASSETS_DIR  = "naruto_parts"      # 파츠 폴더 (anchors.json 포함)

REF_W = REF_H = None
CANVAS_W = CANVAS_H = None

OUT_VIDEO   = "naruto_result.mp4"
OUT_SNAPSHOT= "snapshot_firstframe.png"

SIDE_EXTRA  = 240
STRIDE      = 1
SHOW_DEBUG  = True
FOLLOW_CENTER_X = True
POSE_SWAP_LR= True
POSE_HFLIP  = False

# ---- COCO idx (사용되는 관절만) ----
L_SH=5; R_SH=6; L_EL=7; R_EL=8; L_WR=9; R_WR=10
L_HP=11; R_HP=12; L_KN=13; R_KN=14; L_AN=15; R_AN=16

# ===== 전역: anchors.json 로드 후 채워짐 =====
ANCHORS = {}     # dict[(src,dst)] = {"parent":(x,y), "child":(x,y)}
PARTS   = {}     # dict[name]      = {"file": "filename.png"}
TIP_LOWER = {}   # dict[name]      = (x,y)

# ===== 유틸 =====
def load_rgba_resized(path):
    if REF_W is None or REF_H is None:
        raise RuntimeError("REF_W/REF_H must be set from anchors.json before loading assets.")
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None: raise FileNotFoundError(path)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        a = np.full((*img.shape[:2],1), 255, np.uint8)
        img = np.concatenate([img,a],axis=2)
    if (img.shape[1], img.shape[0]) != (REF_W, REF_H):
        img = cv2.resize(img,(REF_W,REF_H),interpolation=cv2.INTER_LINEAR)
    return img

def H_translate(tx,ty): 
    return np.array([[1,0,tx],[0,1,ty],[0,0,1]],np.float32)

def warp_full(img_rgba,H):
    if CANVAS_W is None or CANVAS_H is None:
        raise RuntimeError("CANVAS_W/H not set.")
    M=H[:2,:]
    return cv2.warpAffine(img_rgba,M,(CANVAS_W,CANVAS_H),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0,0))

def alpha_paste_full(dst,src):
    if src.shape[2]==3:
        a=np.ones((*src.shape[:2],1),np.float32)
        rgb=src.astype(np.float32)
    else:
        a=(src[:,:,3:4].astype(np.float32))/255.0
        rgb=src[:,:,:3].astype(np.float32)
    dst[:]=(dst.astype(np.float32)*(1-a)+rgb*a).astype(np.uint8)
    return dst

def _resize_cover(img, tw, th):
    h, w = img.shape[:2]
    if w == 0 or h == 0:
        raise ValueError("Invalid background image size")
    scale = max(tw / float(w), th / float(h))
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    x0 = max(0, (nw - tw) // 2); y0 = max(0, (nh - th) // 2)
    return resized[y0:y0+th, x0:x0+tw]

def _resize_fit(img, tw, th, pad_color=(0,0,0)):
    """레터박스: 비율 유지 + 여백 패딩."""
    h, w = img.shape[:2]
    scale = min(tw / float(w), th / float(h))
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((th, tw, 3), pad_color, np.uint8)
    x0 = (tw - nw)//2; y0 = (th - nh)//2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas

def _to_bgr3(img):
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 4:
        a = (img[:,:,3:4].astype(np.float32) / 255.0)
        rgb = img[:,:,:3].astype(np.float32)
        base = np.zeros_like(rgb, dtype=np.float32)
        return (base*(1.0-a) + rgb*a).astype(np.uint8)
    if img.ndim == 3 and img.shape[2] == 3:
        return img
    raise ValueError("Unsupported background image format")

def build_background_from_spec(assets_dir, spec, tw, th):
    """
    spec 예:
    {
      "path": "stage.jpg",             # 없으면 단색
      "color": [0,0,0],                # BGR, path 없거나 fit padding용
      "resize": "cover",               # "cover" | "fit"
      "blur": 0                        # 가우시안 블러 커널 근사(픽셀)
    }
    """
    color = tuple(spec.get("color", [0,0,0]))
    mode  = spec.get("resize", "cover")
    blur  = float(spec.get("blur", 0))
    path  = spec.get("path", None)

    if path:
        if not os.path.isabs(path):
            path = os.path.join(assets_dir, path)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Background not found: {path}")
        img3 = _to_bgr3(img)
        if mode == "fit":
            bg = _resize_fit(img3, tw, th, pad_color=color)
        else:
            bg = _resize_cover(img3, tw, th)
    else:
        bg = np.full((th, tw, 3), color, np.uint8)

    if blur > 0:
        k = int(max(1, round(blur)))
        if k % 2 == 0: k += 1
        bg = cv2.GaussianBlur(bg, (k,k), 0)
    return bg

def load_json_scaled(path):
    if REF_W is None or REF_H is None:
        raise RuntimeError("REF_W/REF_H must be set from anchors.json before loading pose JSON.")
    with open(path,"r",encoding="utf-8") as f:
        data=json.load(f)
    fps=float(data.get("fps",30.0))
    stride=int(data.get("stride",1))
    Wv,Hv=data["video_size"]
    sx,sy=REF_W/float(Wv),REF_H/float(Hv)
    frames=[]
    for fr in data["frames"]:
        pts=[]
        for xy in fr["kps"]:
            if xy[0] is None or xy[1] is None:
                pts.append([np.nan,np.nan])
            else:
                pts.append([float(xy[0])*sx,float(xy[1])*sy])
        frames.append(np.array(pts,np.float32))
    return fps/stride, frames

def hip_center_x(kps):
    if np.all(np.isfinite(kps[[L_HP,R_HP]])):
        return 0.5*(kps[L_HP,0]+kps[R_HP,0])
    return np.nan

def swap_lr_labels(kps):
    out=kps.copy()
    pairs=[(L_SH,R_SH),(L_EL,R_EL),(L_WR,R_WR),
           (L_HP,R_HP),(L_KN,R_KN),(L_AN,R_AN)]
    for l,r in pairs: out[[l,r]]=out[[r,l]]
    return out

def hflip_coords(kps):
    out=kps.copy()
    if out.size==0: return out
    out[:,0]=REF_W-1-out[:,0]
    return out

# ===== 스켈레톤: JSON 그대로 사용 =====
def reconstruct_skeleton_follow_json(kps_raw, dx, dy):
    out=np.full((17,2),np.nan,np.float32)
    for i,pt in enumerate(kps_raw):
        if np.all(np.isfinite(pt)):
            out[i]=pt+[dx,dy]
    return out

def compute_body_shrink_ratio(kps_scaled):
    # JSON 어깨폭 / 기준 PNG 어깨폭
    base_L = np.array(ANCHORS[("body","left_upper_arm")]["parent"], np.float32)
    base_R = np.array(ANCHORS[("body","right_upper_arm")]["parent"], np.float32)
    base_shoulder = np.linalg.norm(base_L - base_R) + 1e-6

    if np.all(np.isfinite(kps_scaled[[L_SH, R_SH]])):
        shoulder_width = np.linalg.norm(kps_scaled[L_SH] - kps_scaled[R_SH])
        r = shoulder_width / base_shoulder
        return float(min(1.0, max(0.1, r)))  # 0.1 하한은 안정성용
    return 1.0

def attach_body_affine(assets, canvas, world_LS, world_RS, world_HC, scale_x=1.0):
    if np.any(~np.isfinite([world_LS, world_RS, world_HC])):
        return
    img = assets["body"]

    # 로컬(기준 PNG)에서의 3점
    pL = np.array(ANCHORS[("body","left_upper_arm")]["parent"], np.float32)
    pR = np.array(ANCHORS[("body","right_upper_arm")]["parent"], np.float32)
    pH = 0.5*(np.array(ANCHORS[("left_upper_leg","body")]["parent"], np.float32) +
              np.array(ANCHORS[("right_upper_leg","body")]["parent"], np.float32))
    src_pts = np.array([pL, pR, pH], np.float32)

    # 월드(=JSON)에서의 3점: LS, RS, 힙센터
    dst_pts = np.array([world_LS, world_RS, world_HC], np.float32)

    H = cv2.getAffineTransform(src_pts, dst_pts)
    warped = cv2.warpAffine(img, H, (CANVAS_W, CANVAS_H),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
    alpha_paste_full(canvas, warped)

def get_part_base_width(anchor_start, anchor_end):
    p0 = np.array(anchor_start, np.float32)
    p1 = np.array(anchor_end,   np.float32)
    v = p1 - p0
    length = np.linalg.norm(v)
    if length < 1e-6:
        return 1.0
    return length * 0.1  # 튜닝 가능

def attach_segment_scaled(part_name, assets,
                          anchor_start, anchor_end,
                          world_start, world_end,
                          canvas, scale_x=1.0):
    if np.any(~np.isfinite(world_start)) or np.any(~np.isfinite(world_end)):
        return

    img = assets[part_name]

    # --- 로컬(파츠 PNG) 앵커 2점
    p0 = np.array(anchor_start, np.float32)
    p1 = np.array(anchor_end,   np.float32)
    seg_local = p1 - p0
    len_local = np.linalg.norm(seg_local) + 1e-6
    dir_local = seg_local / len_local
    perp_local = np.array([-dir_local[1], dir_local[0]], np.float32)

    # 두께 기준(로컬) 자동 추출
    base_width = get_part_base_width(anchor_start, anchor_end)

    # 로컬의 세 번째 점: 직선성 방지 + 두께 기준값 반영
    mid_local = 0.5*(p0 + p1) + perp_local * base_width

    # --- 월드(=JSON) 끝점
    q0 = np.array(world_start, np.float32)
    q1 = np.array(world_end,   np.float32)
    seg_world = q1 - q0
    len_world = np.linalg.norm(seg_world)
    if len_world < 1e-6:
        return
    dir_world = seg_world / len_world
    perp_world = np.array([-dir_world[1], dir_world[0]], np.float32)

    # 두께를 body의 폭 축소 비율만큼만 감소
    thickness = base_width * float(min(1.0, max(0.1, scale_x)))

    # 월드의 세 번째 점: 두께 방향으로만 이동
    mid_world = 0.5*(q0 + q1) + perp_world * thickness

    # --- Affine
    src_pts = np.array([p0, p1, mid_local], np.float32)
    dst_pts = np.array([q0, q1, mid_world], np.float32)

    H = cv2.getAffineTransform(src_pts, dst_pts)
    warped = cv2.warpAffine(img, H, (CANVAS_W,CANVAS_H),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
    alpha_paste_full(canvas, warped)

# ===== anchors.json 로더 =====
def load_asset_pack(assets_dir):
    """
    anchors.json을 읽어 전역(ANCHORS, TIP_LOWER, PARTS)과 ref_size, options(=background 포함)를 리턴.
    anchors.json 포맷 예:
    {
      "ref_size": [600, 1000],
      "parts": {"body": "body.png", ...},
      "anchors": {"body>left_upper_arm": {"parent":[x,y], "child":[x,y]}, ...},
      "tip_lower": {"left_lower_arm":[x,y], ...},
      "background": { "path":"stage.jpg", "color":[0,0,0], "resize":"cover", "blur":0 },
      "options": {...}  # (있어도, 없어도 됨)
    }
    """
    cfg_path = os.path.join(assets_dir, "anchors.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"anchors.json이 없습니다: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    ref_size = cfg.get("ref_size")
    if not ref_size:
        raise ValueError("anchors.json: 'ref_size'가 필요합니다. 예: [600, 1000]")
    ref_w, ref_h = int(ref_size[0]), int(ref_size[1])

    parts_json = cfg.get("parts", {})
    if not parts_json:
        raise ValueError("anchors.json: 'parts'가 비었습니다.")

    # PARTS
    parts = {k: {"file": v} for k, v in parts_json.items()}

    # ANCHORS
    anchors = {}
    anchors_json = cfg.get("anchors", {})
    for key, meta in anchors_json.items():
        if ">" not in key: 
            continue
        a,b = key.split(">",1)
        parent = (float(meta["parent"][0]), float(meta["parent"][1]))
        child  = (float(meta["child"][0]),  float(meta["child"][1]))
        anchors[(a,b)] = {"parent": parent, "child": child}

    # TIP_LOWER
    tips = {}
    tips_json = cfg.get("tip_lower", {})
    for name, pt in tips_json.items():
        tips[name] = (float(pt[0]), float(pt[1]))

    # background spec (옵션)
    background = cfg.get("background", {}) or {}

    options = cfg.get("options", {})
    options["background"] = background  # options에 포함시켜 전달

    # 간단 검증
    required_anchors = [
        ("body","left_upper_arm"),
        ("body","right_upper_arm"),
        ("left_upper_arm","left_lower_arm"),
        ("right_upper_arm","right_lower_arm"),
        ("left_upper_leg","body"),
        ("right_upper_leg","body"),
        ("left_upper_leg","left_lower_leg"),
        ("right_upper_leg","right_lower_leg"),
    ]
    missing = [k for k in required_anchors if k not in anchors]
    required_tips = ["left_lower_arm","right_lower_arm","left_lower_leg","right_lower_leg"]
    missing += [("tip",k) for k in required_tips if k not in tips]
    if missing:
        raise ValueError(f"anchors.json 필수 키 누락: {missing}")

    return anchors, tips, parts, ref_w, ref_h, options

# ===== 한 프레임 렌더 =====
def render_pose_frame(kps_raw, assets, offset_x, top_pad, debug=False, background=None):
    dx = float(offset_x)
    if FOLLOW_CENTER_X:
        hx = hip_center_x(kps_raw)
        if np.isfinite(hx):
            dx += CANVAS_W * 0.5 - float(hx)
    dy = float(top_pad)

    kps_scaled = reconstruct_skeleton_follow_json(kps_raw, dx, dy)

    # 배경 초기화
    if background is not None:
        canvas = background.copy()
    else:
        canvas = np.zeros((CANVAS_H, CANVAS_W, 3), np.uint8)

    # === body shrink ratio (어깨폭 기준) ===
    scale_x = compute_body_shrink_ratio(kps_scaled)

    # [LAYER 1] 다리
    if np.all(np.isfinite(kps_scaled[[L_HP, L_KN]])):
        attach_segment_scaled("left_upper_leg", assets,
            ANCHORS[("left_upper_leg","body")]["parent"],
            ANCHORS[("left_upper_leg","left_lower_leg")]["parent"],
            kps_scaled[L_HP], kps_scaled[L_KN], canvas, scale_x=scale_x)
    if np.all(np.isfinite(kps_scaled[[L_KN, L_AN]])):
        attach_segment_scaled("left_lower_leg", assets,
            ANCHORS[("left_upper_leg","left_lower_leg")]["child"],
            TIP_LOWER["left_lower_leg"],
            kps_scaled[L_KN], kps_scaled[L_AN], canvas, scale_x=scale_x)

    if np.all(np.isfinite(kps_scaled[[R_HP, R_KN]])):
        attach_segment_scaled("right_upper_leg", assets,
            ANCHORS[("right_upper_leg","body")]["parent"],
            ANCHORS[("right_upper_leg","right_lower_leg")]["parent"],
            kps_scaled[R_HP], kps_scaled[R_KN], canvas, scale_x=scale_x)
    if np.all(np.isfinite(kps_scaled[[R_KN, R_AN]])):
        attach_segment_scaled("right_lower_leg", assets,
            ANCHORS[("right_upper_leg","right_lower_leg")]["child"],
            TIP_LOWER["right_lower_leg"],
            kps_scaled[R_KN], kps_scaled[R_AN], canvas, scale_x=scale_x)

    # [LAYER 2] 몸통
    if np.all(np.isfinite(kps_scaled[[L_SH, R_SH, L_HP, R_HP]])):
        hip_center = 0.5 * (kps_scaled[L_HP] + kps_scaled[R_HP])
        attach_body_affine(assets, canvas,
                           kps_scaled[L_SH], kps_scaled[R_SH], hip_center,
                           scale_x=scale_x)
    else:
        alpha_paste_full(canvas, warp_full(assets["body"], H_translate(dx, dy)))

    # 윗팔
    if np.all(np.isfinite(kps_scaled[[L_SH, L_EL]])):
        attach_segment_scaled("left_upper_arm", assets,
            ANCHORS[("body","left_upper_arm")]["child"],
            ANCHORS[("left_upper_arm","left_lower_arm")]["parent"],
            kps_scaled[L_SH], kps_scaled[L_EL], canvas, scale_x=scale_x)
    if np.all(np.isfinite(kps_scaled[[R_SH, R_EL]])):
        attach_segment_scaled("right_upper_arm", assets,
            ANCHORS[("body","right_upper_arm")]["child"],
            ANCHORS[("right_upper_arm","right_lower_arm")]["parent"],
            kps_scaled[R_SH], kps_scaled[R_EL], canvas, scale_x=scale_x)

    # [LAYER 3] 아래팔/손
    if np.all(np.isfinite(kps_scaled[[L_EL, L_WR]])):
        attach_segment_scaled("left_lower_arm", assets,
            ANCHORS[("left_upper_arm","left_lower_arm")]["child"],
            TIP_LOWER["left_lower_arm"],
            kps_scaled[L_EL], kps_scaled[L_WR], canvas, scale_x=scale_x)
    if np.all(np.isfinite(kps_scaled[[R_EL, R_WR]])):
        attach_segment_scaled("right_lower_arm", assets,
            ANCHORS[("right_upper_arm","right_lower_arm")]["child"],
            TIP_LOWER["right_lower_arm"],
            kps_scaled[R_EL], kps_scaled[R_WR], canvas, scale_x=scale_x)

    # 디버그 점
    if SHOW_DEBUG:
        for i in range(len(kps_scaled)):
            if np.all(np.isfinite(kps_scaled[i])):
                cv2.circle(canvas, tuple(np.int32(kps_scaled[i])), 3, (0,255,255), -1)

    return canvas

# ================= MAIN =================
def main():
    global REF_W, REF_H, CANVAS_W, CANVAS_H, ANCHORS, PARTS, TIP_LOWER

    # 1) anchors.json 로드 (ref_size & background 포함)
    ANCHORS, TIP_LOWER, PARTS, ref_w, ref_h, options = load_asset_pack(ASSETS_DIR)

    # 2) 기준 해상도 업데이트
    REF_W, REF_H = int(ref_w), int(ref_h)

    # 3) 캔버스 크기 확정
    CANVAS_W = REF_W + SIDE_EXTRA*2
    CANVAS_H = REF_H + 400

    # 4) 배경(spec)로부터 배경 이미지 빌드
    bg_spec = options.get("background", {}) or {}
    background = build_background_from_spec(ASSETS_DIR, bg_spec, CANVAS_W, CANVAS_H)

    # 5) 파츠 이미지 로드/리사이즈
    assets={name:load_rgba_resized(os.path.join(ASSETS_DIR,meta["file"])) for name,meta in PARTS.items()}

    # 6) 포즈 JSON 로드
    fps,frames=load_json_scaled(JSON_PATH)
    print(f"[info] frames={len(frames)} fps={fps:.3f}")

    # 7) 비디오 라이터
    writer=cv2.VideoWriter(OUT_VIDEO,cv2.VideoWriter_fourcc(*"mp4v"),fps,(CANVAS_W,CANVAS_H))

    # 8) 렌더 루프
    for i in range(0,len(frames),STRIDE):
        k=frames[i].copy()
        if POSE_HFLIP: k=hflip_coords(k)
        if POSE_SWAP_LR: k=swap_lr_labels(k)
        img=render_pose_frame(k,assets,0,100,debug=SHOW_DEBUG, background=background)
        if i==0: cv2.imwrite(OUT_SNAPSHOT,img)
        writer.write(img)

    writer.release()
    print(f"[ok] saved: {OUT_VIDEO}, {OUT_SNAPSHOT}")

if __name__=="__main__":
    main()
