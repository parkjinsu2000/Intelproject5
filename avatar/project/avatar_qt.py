
import os, json
import numpy as np

# ✅ PyQt 먼저
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtGui import QImage

# ✅ 그 다음 OpenCV
import cv2


# ---- COCO indices (used joints) ----
L_SH=5; R_SH=6; L_EL=7; R_EL=8; L_WR=9; R_WR=10
L_HP=11; R_HP=12; L_KN=13; R_KN=14; L_AN=15; R_AN=16


class MannequinRenderer(QObject):
    # ===== Qt Signals =====
    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    error = pyqtSignal(str)
    playReady = pyqtSignal(object, float)  # (frames: list[QImage], fps: float)
    finished = pyqtSignal()  

    def __init__(
        self,
        json_path: str,
        assets_dir: str,
        side_extra: int = 240,
        top_pad: int = 100,
        stride: int = 1,
        show_debug: bool = True,
        follow_center_x: bool = True,
        pose_swap_lr: bool = True,
        pose_hflip: bool = False,
        parent=None
    ):
        super().__init__(parent)
        self.json_path = json_path
        self.assets_dir = assets_dir

        self.side_extra = int(side_extra)
        self.top_pad = int(top_pad)
        self.stride = int(stride)

        self.show_debug = bool(show_debug)
        self.follow_center_x = bool(follow_center_x)
        self.pose_swap_lr = bool(pose_swap_lr)
        self.pose_hflip = bool(pose_hflip)

        # Filled by anchors.json
        self.REF_W = None
        self.REF_H = None
        self.CANVAS_W = None
        self.CANVAS_H = None

        self.ANCHORS = {}
        self.PARTS = {}
        self.TIP_LOWER = {}
        self.options = {}

        # Cached assets/background
        self.assets = {}
        self.background = None
        self.v_align_mode = "center"
        self._cancel = False 


    def cancel(self):                    # ✅ 외부에서 취소 요청
        self._cancel = True

    # ===================== Utility =====================

    @staticmethod
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
        raise ValueError("Unsupported image format")

    @staticmethod
    def _resize_cover(img, tw, th):
        h, w = img.shape[:2]
        if w == 0 or h == 0:
            raise ValueError("Invalid image size")
        scale = max(tw / float(w), th / float(h))
        nw, nh = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        x0 = max(0, (nw - tw) // 2); y0 = max(0, (nh - th) // 2)
        return resized[y0:y0+th, x0:x0+tw]

    @staticmethod
    def _resize_fit(img, tw, th, pad_color=(0,0,0)):
        h, w = img.shape[:2]
        scale = min(tw / float(w), th / float(h))
        nw, nh = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((th, tw, 3), pad_color, np.uint8)
        x0 = (tw - nw)//2; y0 = (th - nh)//2
        canvas[y0:y0+nh, x0:x0+nw] = resized
        return canvas

    @staticmethod
    def _cv_bgr_to_qimage(img_bgr):
        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        qimg = QImage(img_rgb.data, w, h, img_rgb.strides[0], QImage.Format_RGB888)
        return qimg.copy()
    
    def _compute_dy(self, kps_raw, top_pad):
        mode = getattr(self, "v_align_mode", "top_pad")
        H = self.CANVAS_H

        ys = [float(kps_raw[i,1]) for i in range(kps_raw.shape[0]) if np.isfinite(kps_raw[i,1])]
        if not ys:
            return float(top_pad)

        ymin, ymax = min(ys), max(ys)

        if mode == "center":
            char_h = max(1.0, ymax - ymin)
            target_top = (H - char_h) * 0.5
            return float(target_top - ymin)

        if mode == "feet":
            cand = []
            for idx in (L_AN, R_AN, L_KN, R_KN, L_HP, R_HP):
                y = kps_raw[idx,1]
                if np.isfinite(y):
                    cand.append(float(y))
            anchor_y = max(cand) if cand else ymax
            floor_y = H - float(getattr(self, "bottom_margin_px", 40))
            return float(floor_y - anchor_y)

        return float(top_pad)

    # ===================== Loaders =====================

    def load_asset_pack(self):
        cfg_path = os.path.join(self.assets_dir, "anchors.json")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"anchors.json이 없습니다: {cfg_path}")

        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        ref_size = cfg.get("ref_size")
        if not ref_size:
            raise ValueError("anchors.json: 'ref_size'가 필요합니다. 예: [600, 1000]")
        self.REF_W, self.REF_H = int(ref_size[0]), int(ref_size[1])

        parts_json = cfg.get("parts", {})
        if not parts_json:
            raise ValueError("anchors.json: 'parts'가 비었습니다.")
        self.PARTS = {k: {"file": v} for k, v in parts_json.items()}

        # anchors
        self.ANCHORS = {}
        anchors_json = cfg.get("anchors", {})
        for key, meta in anchors_json.items():
            if ">" not in key:
                continue
            a,b = key.split(">",1)
            parent = (float(meta["parent"][0]), float(meta["parent"][1]))
            child  = (float(meta["child"][0]),  float(meta["child"][1]))
            self.ANCHORS[(a,b)] = {"parent": parent, "child": child}

        # tip_lower
        self.TIP_LOWER = {}
        tips_json = cfg.get("tip_lower", {})
        for name, pt in tips_json.items():
            self.TIP_LOWER[name] = (float(pt[0]), float(pt[1]))

        # background spec
        self.options = cfg.get("options", {}) or {}
        self.options["background"] = cfg.get("background", {}) or {}

        # quick validation
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
        missing = [k for k in required_anchors if k not in self.ANCHORS]
        required_tips = ["left_lower_arm","right_lower_arm","left_lower_leg","right_lower_leg"]
        missing += [("tip",k) for k in required_tips if k not in self.TIP_LOWER]
        if missing:
            raise ValueError(f"anchors.json 필수 키 누락: {missing}")

    def load_rgba_resized(self, path):
        if self.REF_W is None or self.REF_H is None:
            raise RuntimeError("REF_W/REF_H must be set before loading assets.")
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(path)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        elif img.shape[2] == 3:
            a = np.full((*img.shape[:2],1), 255, np.uint8)
            img = np.concatenate([img,a],axis=2)
        if (img.shape[1], img.shape[0]) != (self.REF_W, self.REF_H):
            img = cv2.resize(img,(self.REF_W,self.REF_H),interpolation=cv2.INTER_LINEAR)
        return img

    def build_background_from_spec(self, spec, tw, th):
        color = tuple(spec.get("color", [0,0,0]))
        mode  = spec.get("resize", "cover")
        blur  = float(spec.get("blur", 0))
        path  = spec.get("path", None)

        if path:
            if not os.path.isabs(path):
                path = os.path.join(self.assets_dir, path)
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise FileNotFoundError(f"Background not found: {path}")
            img3 = self._to_bgr3(img)
            if mode == "native":
                bg = img3  # 사이즈 변경 없음
            elif mode == "fit":
                bg = self._resize_fit(img3, tw, th, pad_color=color)
            else:
                bg = self._resize_cover(img3, tw, th)
        else:
            bg = np.full((th, tw, 3), color, np.uint8)

        if blur > 0:
            k = int(max(1, round(blur)))
            if k % 2 == 0: k += 1
            bg = cv2.GaussianBlur(bg, (k,k), 0)
        return bg

    def load_json_scaled(self, path):
        if self.REF_W is None or self.REF_H is None:
            raise RuntimeError("REF_W/REF_H must be set before loading pose JSON.")
        with open(path,"r",encoding="utf-8") as f:
            data=json.load(f)
        fps=float(data.get("fps",30.0))
        stride=int(data.get("stride",1))
        Wv,Hv=data["video_size"]
        sx,sy=self.REF_W/float(Wv),self.REF_H/float(Hv)
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

    # ===================== Math / Render helpers =====================

    @staticmethod
    def H_translate(tx,ty):
        return np.array([[1,0,tx],[0,1,ty],[0,0,1]],np.float32)

    def warp_full(self, img_rgba, H):
        M=H[:2,:]
        return cv2.warpAffine(
            img_rgba, M, (self.CANVAS_W,self.CANVAS_H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0)
        )

    @staticmethod
    def alpha_paste_full(dst,src):
        if src.shape[2]==3:
            a=np.ones((*src.shape[:2],1),np.float32)
            rgb=src.astype(np.float32)
        else:
            a=(src[:,:,3:4].astype(np.float32))/255.0
            rgb=src[:,:,:3].astype(np.float32)
        dst[:]=(dst.astype(np.float32)*(1-a)+rgb*a).astype(np.uint8)
        return dst

    @staticmethod
    def hip_center_x(kps):
        if np.all(np.isfinite(kps[[L_HP,R_HP]])):
            return 0.5*(kps[L_HP,0]+kps[R_HP,0])
        return np.nan

    @staticmethod
    def swap_lr_labels(kps):
        out=kps.copy()
        pairs=[(L_SH,R_SH),(L_EL,R_EL),(L_WR,R_WR),
               (L_HP,R_HP),(L_KN,R_KN),(L_AN,R_AN)]
        for l,r in pairs: out[[l,r]]=out[[r,l]]
        return out

    def hflip_coords(self, kps):
        out=kps.copy()
        if out.size==0: return out
        out[:,0]=self.REF_W-1-out[:,0]
        return out

    @staticmethod
    def reconstruct_skeleton_follow_json(kps_raw, dx, dy):
        out=np.full((17,2),np.nan,np.float32)
        for i,pt in enumerate(kps_raw):
            if np.all(np.isfinite(pt)):
                out[i]=pt+[dx,dy]
        return out

    def compute_body_shrink_ratio(self, kps_scaled):
        base_L = np.array(self.ANCHORS[("body","left_upper_arm")]["parent"], np.float32)
        base_R = np.array(self.ANCHORS[("body","right_upper_arm")]["parent"], np.float32)
        base_shoulder = np.linalg.norm(base_L - base_R) + 1e-6

        if np.all(np.isfinite(kps_scaled[[L_SH, R_SH]])):
            shoulder_width = np.linalg.norm(kps_scaled[L_SH] - kps_scaled[R_SH])
            r = shoulder_width / base_shoulder
            return float(min(1.0, max(0.1, r)))
        return 1.0

    @staticmethod
    def get_part_base_width(anchor_start, anchor_end):
        p0 = np.array(anchor_start, np.float32)
        p1 = np.array(anchor_end,   np.float32)
        v = p1 - p0
        length = np.linalg.norm(v)
        if length < 1e-6:
            return 1.0
        return length * 0.1

    def attach_body_affine(self, assets, canvas, world_LS, world_RS, world_HC, scale_x=1.0):
        if np.any(~np.isfinite([world_LS, world_RS, world_HC])):
            return
        img = assets["body"]

        pL = np.array(self.ANCHORS[("body","left_upper_arm")]["parent"], np.float32)
        pR = np.array(self.ANCHORS[("body","right_upper_arm")]["parent"], np.float32)
        pH = 0.5*(np.array(self.ANCHORS[("left_upper_leg","body")]["parent"], np.float32) +
                  np.array(self.ANCHORS[("right_upper_leg","body")]["parent"], np.float32))
        src_pts = np.array([pL, pR, pH], np.float32)
        dst_pts = np.array([world_LS, world_RS, world_HC], np.float32)

        H = cv2.getAffineTransform(src_pts, dst_pts)
        warped = cv2.warpAffine(img, H, (self.CANVAS_W, self.CANVAS_H),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
        self.alpha_paste_full(canvas, warped)

    def attach_segment_scaled(self, part_name, assets,
                              anchor_start, anchor_end,
                              world_start, world_end,
                              canvas, scale_x=1.0):
        if np.any(~np.isfinite(world_start)) or np.any(~np.isfinite(world_end)):
            return

        img = assets[part_name]

        # local anchors
        p0 = np.array(anchor_start, np.float32)
        p1 = np.array(anchor_end,   np.float32)
        seg_local = p1 - p0
        len_local = np.linalg.norm(seg_local) + 1e-6
        dir_local = seg_local / len_local
        perp_local = np.array([-dir_local[1], dir_local[0]], np.float32)

        base_width = self.get_part_base_width(anchor_start, anchor_end)
        mid_local = 0.5*(p0 + p1) + perp_local * base_width

        # world segment
        q0 = np.array(world_start, np.float32)
        q1 = np.array(world_end,   np.float32)
        seg_world = q1 - q0
        len_world = np.linalg.norm(seg_world)
        if len_world < 1e-6:
            return
        dir_world = seg_world / len_world
        perp_world = np.array([-dir_world[1], dir_world[0]], np.float32)

        thickness = base_width * float(min(1.0, max(0.1, scale_x)))
        mid_world = 0.5*(q0 + q1) + perp_world * thickness

        src_pts = np.array([p0, p1, mid_local], np.float32)
        dst_pts = np.array([q0, q1, mid_world], np.float32)

        H = cv2.getAffineTransform(src_pts, dst_pts)
        warped = cv2.warpAffine(img, H, (self.CANVAS_W,self.CANVAS_H),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
        self.alpha_paste_full(canvas, warped)

    # ===================== Frame render =====================

    def render_pose_frame(self, kps_raw, assets, offset_x, top_pad, background=None):
        dx = float(offset_x)
        if self.follow_center_x:
            hx = self.hip_center_x(kps_raw)
            if np.isfinite(hx):
                dx += self.CANVAS_W * 0.5 - float(hx)
        dy = self._compute_dy(kps_raw, top_pad)

        kps_scaled = self.reconstruct_skeleton_follow_json(kps_raw, dx, dy)

        # background
        if background is not None:
            canvas = background.copy()
        else:
            canvas = np.zeros((self.CANVAS_H, self.CANVAS_W, 3), np.uint8)

        # body thickness ratio by shoulders
        scale_x = self.compute_body_shrink_ratio(kps_scaled)

        # 유효성 플래그
        left_upper_ok  = np.all(np.isfinite(kps_scaled[[L_HP, L_KN]]))
        left_lower_ok  = np.all(np.isfinite(kps_scaled[[L_KN, L_AN]]))
        right_upper_ok = np.all(np.isfinite(kps_scaled[[R_HP, R_KN]]))
        right_lower_ok = np.all(np.isfinite(kps_scaled[[R_KN, R_AN]]))
        body_ok        = np.all(np.isfinite(kps_scaled[[L_SH, R_SH, L_HP, R_HP]]))

        # ren_parts 모드 감지: 폴더명이 ren_parts / rens_parts 인 경우
        ren_mode = any(s in os.path.basename(self.assets_dir).lower() for s in ("ren_parts", "rens_parts"))

        if ren_mode:
            # ===== ren_parts: 몸통 먼저 → 다리(몸통 위로) → 팔 =====
            # [LAYER 1] 몸통
            if body_ok:
                hip_center = 0.5 * (kps_scaled[L_HP] + kps_scaled[R_HP])
                self.attach_body_affine(assets, canvas,
                                        kps_scaled[L_SH], kps_scaled[R_SH], hip_center,
                                        scale_x=scale_x)
            else:
                self.alpha_paste_full(canvas, self.warp_full(assets["body"], self.H_translate(dx, dy)))

            # [LAYER 2] 다리 (몸통 앞쪽에 나오도록)
            if left_upper_ok:
                self.attach_segment_scaled("left_upper_leg", assets,
                    self.ANCHORS[("left_upper_leg","body")]["parent"],
                    self.ANCHORS[("left_upper_leg","left_lower_leg")]["parent"],
                    kps_scaled[L_HP], kps_scaled[L_KN], canvas, scale_x=scale_x)
            if left_lower_ok:
                self.attach_segment_scaled("left_lower_leg", assets,
                    self.ANCHORS[("left_upper_leg","left_lower_leg")]["child"],
                    self.TIP_LOWER["left_lower_leg"],
                    kps_scaled[L_KN], kps_scaled[L_AN], canvas, scale_x=scale_x)

            if right_upper_ok:
                self.attach_segment_scaled("right_upper_leg", assets,
                    self.ANCHORS[("right_upper_leg","body")]["parent"],
                    self.ANCHORS[("right_upper_leg","right_lower_leg")]["parent"],
                    kps_scaled[R_HP], kps_scaled[R_KN], canvas, scale_x=scale_x)
            if right_lower_ok:
                self.attach_segment_scaled("right_lower_leg", assets,
                    self.ANCHORS[("right_upper_leg","right_lower_leg")]["child"],
                    self.TIP_LOWER["right_lower_leg"],
                    kps_scaled[R_KN], kps_scaled[R_AN], canvas, scale_x=scale_x)

        else:
            # ===== 기본: 다리 → 몸통 → 팔 (기존 동작 유지) =====
            # [LAYER 1] 다리
            if left_upper_ok:
                self.attach_segment_scaled("left_upper_leg", assets,
                    self.ANCHORS[("left_upper_leg","body")]["parent"],
                    self.ANCHORS[("left_upper_leg","left_lower_leg")]["parent"],
                    kps_scaled[L_HP], kps_scaled[L_KN], canvas, scale_x=scale_x)
            if left_lower_ok:
                self.attach_segment_scaled("left_lower_leg", assets,
                    self.ANCHORS[("left_upper_leg","left_lower_leg")]["child"],
                    self.TIP_LOWER["left_lower_leg"],
                    kps_scaled[L_KN], kps_scaled[L_AN], canvas, scale_x=scale_x)

            if right_upper_ok:
                self.attach_segment_scaled("right_upper_leg", assets,
                    self.ANCHORS[("right_upper_leg","body")]["parent"],
                    self.ANCHORS[("right_upper_leg","right_lower_leg")]["parent"],
                    kps_scaled[R_HP], kps_scaled[R_KN], canvas, scale_x=scale_x)
            if right_lower_ok:
                self.attach_segment_scaled("right_lower_leg", assets,
                    self.ANCHORS[("right_upper_leg","right_lower_leg")]["child"],
                    self.TIP_LOWER["right_lower_leg"],
                    kps_scaled[R_KN], kps_scaled[R_AN], canvas, scale_x=scale_x)

            # [LAYER 2] 몸통
            if body_ok:
                hip_center = 0.5 * (kps_scaled[L_HP] + kps_scaled[R_HP])
                self.attach_body_affine(assets, canvas,
                                        kps_scaled[L_SH], kps_scaled[R_SH], hip_center,
                                        scale_x=scale_x)
            else:
                self.alpha_paste_full(canvas, self.warp_full(assets["body"], self.H_translate(dx, dy)))

        # [LAYER 3] 팔 (두 모드 모두 최상단)
        if np.all(np.isfinite(kps_scaled[[L_SH, L_EL]])):
            self.attach_segment_scaled("left_upper_arm", assets,
                self.ANCHORS[("body","left_upper_arm")]["child"],
                self.ANCHORS[("left_upper_arm","left_lower_arm")]["parent"],
                kps_scaled[L_SH], kps_scaled[L_EL], canvas, scale_x=scale_x)
        if np.all(np.isfinite(kps_scaled[[R_SH, R_EL]])):
            self.attach_segment_scaled("right_upper_arm", assets,
                self.ANCHORS[("body","right_upper_arm")]["child"],
                self.ANCHORS[("right_upper_arm","right_lower_arm")]["parent"],
                kps_scaled[R_SH], kps_scaled[R_EL], canvas, scale_x=scale_x)

        if np.all(np.isfinite(kps_scaled[[L_EL, L_WR]])):
            self.attach_segment_scaled("left_lower_arm", assets,
                self.ANCHORS[("left_upper_arm","left_lower_arm")]["child"],
                self.TIP_LOWER["left_lower_arm"],
                kps_scaled[L_EL], kps_scaled[L_WR], canvas, scale_x=scale_x)
        if np.all(np.isfinite(kps_scaled[[R_EL, R_WR]])):
            self.attach_segment_scaled("right_lower_arm", assets,
                self.ANCHORS[("right_upper_arm","right_lower_arm")]["child"],
                self.TIP_LOWER["right_lower_arm"],
                kps_scaled[R_EL], kps_scaled[R_WR], canvas, scale_x=scale_x)

        # debug joints
        if self.show_debug:
            for i in range(len(kps_scaled)):
                if np.all(np.isfinite(kps_scaled[i])):
                    cv2.circle(canvas, tuple(np.int32(kps_scaled[i])), 3, (0,255,255), -1)

        return canvas

    # ===================== Public API =====================

    def run(self):
        """스레드에서 호출: anchors.json/배경/파츠/포즈 로드 → 모든 프레임 렌더 → playReady(frames, fps)"""
        try:
            self.log.emit("[info] loading anchors.json ...")
            self.load_asset_pack()

            bg_spec = self.options.get("background", {}) or {}
            mode = bg_spec.get("resize", "cover")
            bg_path = bg_spec.get("path", None)

            if mode == "native" and bg_path:  # ✅ 배경 원본 크기에 맞추는 모드
                path = bg_path if os.path.isabs(bg_path) else os.path.join(self.assets_dir, bg_path)
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if img is None:
                    raise FileNotFoundError(f"Background not found: {path}")
                img3 = self._to_bgr3(img)
                self.CANVAS_H, self.CANVAS_W = img3.shape[:2]  # ✅ 캔버스 = 배경 원본 크기
            else:
                # 기존 로직(avatar 크기 기반)
                self.CANVAS_W = self.REF_W + self.side_extra*2
                self.CANVAS_H = self.REF_H + max(0, self.top_pad)

            # 배경 생성 (위에서 정한 캔버스 사이즈로)
            self.background = self.build_background_from_spec(bg_spec, self.CANVAS_W, self.CANVAS_H)

            # 이하 동일 (파츠 로드/프레임 렌더/emit 등) ...

            # parts
            self.assets = {}
            for name, meta in self.PARTS.items():
                p = os.path.join(self.assets_dir, meta["file"])
                self.assets[name] = self.load_rgba_resized(p)

            # pose json
            fps, frames = self.load_json_scaled(self.json_path)
            self.log.emit(f"[info] frames={len(frames)} fps={fps:.3f}")

            # render all frames to QImage list
            qframes = []
            total_steps = len(list(range(0, len(frames), self.stride))) or 1

            for idx, i in enumerate(range(0, len(frames), self.stride), start=1):
                if self._cancel:
                    break
                k = frames[i].copy()
                if self.pose_hflip:
                    k = self.hflip_coords(k)
                if self.pose_swap_lr:
                    k = self.swap_lr_labels(k)

                img = self.render_pose_frame(k, self.assets, 0, self.top_pad, background=self.background)
                qframes.append(self._cv_bgr_to_qimage(img))

                self.progress.emit(int(idx * 100 / total_steps))

            # hand off to UI for playback
            self.playReady.emit(qframes, fps)

        except Exception as e:
            self.error.emit(str(e))

        finally:
            self.finished.emit()
