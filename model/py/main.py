
import argparse
import time
import collections
import numpy as np
import cv2
import torch
from ultralytics import YOLO

from settings import (
    MODEL_PATH_DEFAULT, DETECT_CONF_THRES, SMOOTHING_WINDOW_SIZE, PERSON_COLORS
)
from pose_utils import (
    normalize_keypoints, pose_to_anglevec, frame_score_strict
)
from drawing_utils import (
    draw_pose_with_id, put_text, draw_countdown_overlay
)
from inference import (
    track_once, preprocess_reference_tracks
)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("--source",type=str,default="0")
    ap.add_argument("--start",type=float,default=3.0)
    ap.add_argument("--every",type=int,default=5)
    ap.add_argument("--tracker",type=str,default="py/botsort_reid.yaml")
    ap.add_argument("--model",type=str,default=MODEL_PATH_DEFAULT)
    ap.add_argument("--disp-scale",type=float,default=1.0)
    ap.add_argument("--disp-width",type=int,default=None)
    ap.add_argument("--save",type=str,default=None)
    ap.add_argument("--imgsz",type=int,default=320)
    ap.add_argument("--device",type=str,default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--half",action="store_true",default=False)
    ap.add_argument("--no-mirror",action="store_true")
    ap.add_argument("--force-preprocess",action="store_true")
    ap.add_argument("--max-persons",type=int, default=2, help="Maximum number of people to track")
    args=ap.parse_args()

    mirror=not args.no_mirror
    cv2.setUseOptimized(True); cv2.setNumThreads(0)

    try: cam_idx=int(args.source); src_is_cam=True
    except ValueError: cam_idx=None; src_is_cam=False

    capU = cv2.VideoCapture(cam_idx if src_is_cam else args.source)
    if not capU.isOpened():
        raise RuntimeError(f"Cannot open source: {args.source}")
    capR = cv2.VideoCapture(args.ref)
    if not capR.isOpened():
        raise RuntimeError(f"Cannot open reference: {args.ref}")

    # 윈도 크기
    W = int(max(capR.get(cv2.CAP_PROP_FRAME_WIDTH),  640))
    H = int(max(capR.get(cv2.CAP_PROP_FRAME_HEIGHT), 360))
    size_single = (W, H)

    # 모델
    model = YOLO(args.model)
    try:
        model.to(args.device)
    except Exception:
        pass
    use_half = bool(args.half and args.device=="cuda")
    if use_half:
        try: model.model.half()
        except Exception: use_half = False

    # 워밍업
    _dummy = np.zeros((384, 384, 3), np.uint8)
    with torch.inference_mode():
        for _ in range(2):
            _ = model.predict(_dummy, imgsz=args.imgsz, device=args.device, half=args.half,
                                conf=DETECT_CONF_THRES, verbose=False)[0]

    # 참조 영상 트랙 전처리
    ref_tracks_all_frames = preprocess_reference_tracks(args.ref, model, args.tracker, args.imgsz, args.device, use_half, args.force_preprocess)
    num_ref_frames = len(ref_tracks_all_frames)

    # 저장기
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, int(round(capR.get(cv2.CAP_PROP_FPS) or 30.0)), (W*2, H))

    # ---------------- 카운트다운 (카메라 소스일 때만) ----------------
    if src_is_cam:
        okR0, ref_first = capR.read()
        if not okR0:
            raise RuntimeError("Reference has no frames.")
        ref_first = cv2.resize(ref_first, size_single)

        t0 = time.monotonic()
        countdown = max(0.0, float(args.start))
        print(f"[info] countdown {countdown:.1f}s ...")
        while True:
            now = time.monotonic()
            remain = countdown - (now - t0)
            if remain <= -0.2:
                break
            okU, fU = capU.read()
            if not okU: continue
            fU = cv2.resize(fU, size_single)
            left = ref_first.copy()
            right = fU.copy()
            text = str(int(np.ceil(remain))) if remain > 0.8 else "START"
            draw_countdown_overlay(left, text)
            draw_countdown_overlay(right, text)
            put_text(left, "REF", (12,48), 1.1)
            put_text(right, f"USER [{'mirror' if mirror else 'normal'}]", (12,48), 1.0)
            canvas = np.hstack([left, right])

            disp = canvas
            scale = args.disp_scale
            if args.disp_width is not None and args.disp_width > 0:
                scale = min(scale, float(args.disp_width) / float(canvas.shape[1]))
            if scale < 1.0:
                new_w = int(canvas.shape[1] * scale); new_h = int(canvas.shape[0] * scale)
                disp = cv2.resize(canvas, (new_w, new_h), interpolation=cv2.INTER_AREA)

            cv2.imshow("Multi Dance Compare (q to quit)", disp)
            if writer is not None: writer.write(canvas)
            if (cv2.waitKey(1) & 0xFF)==ord('q'):
                if writer is not None: writer.release()
                capU.release(); capR.release(); cv2.destroyAllWindows(); return

    capR.set(cv2.CAP_PROP_POS_FRAMES, 0)

    id_total = collections.defaultdict(lambda: 100)
    user_vec_histories = collections.defaultdict(lambda: collections.deque(maxlen=SMOOTHING_WINDOW_SIZE))

    frame_idx = 0
    print("[info] scoring start!")
    while True:
        loop_start_time = time.monotonic()

        okR, fR = capR.read()
        okU, fU = capU.read()

        if not okR or not okU:
            if not okR: print("[info] reference video ended.")
            if not okU: print("[warn] source ended or camera failed.")
            break

        fR = cv2.resize(fR, size_single)
        fU = cv2.resize(fU, size_single)

        # 사용자 영상에 대해서만 실시간 트래킹 수행
        if mirror:
            fU = cv2.flip(fU, 1)
        tracks_U = track_once(model, fU, args.tracker, args.imgsz, args.device, use_half, max_persons=args.max_persons)

        # 전처리된 참조 영상 트랙 데이터 가져오기
        tracks_R = {}
        if frame_idx < num_ref_frames:
            tracks_R = ref_tracks_all_frames[frame_idx]
        
        vecR = None
        if tracks_R:
            try:
                ref_tid = min(tracks_R.keys())
                kps_r, _, _ = tracks_R[ref_tid]
                if np.sum(np.isfinite(kps_r)) > 10:
                    kps_n_r = normalize_keypoints(kps_r)
                    vecR = pose_to_anglevec(kps_n_r)
            except (ValueError, KeyError):
                pass

        if vecR is not None and frame_idx % max(1, args.every) == 0 and len(tracks_U) > 0:
            for tid, (kps, conf, _box) in tracks_U.items():
                num_finite_kps = np.sum(np.isfinite(kps))
                if num_finite_kps < 10: continue

                kps_n = normalize_keypoints(kps)
                vecU = pose_to_anglevec(kps_n)
                
                user_vec_histories[tid].append(vecU)
                vecU_smoothed = np.mean(list(user_vec_histories[tid]), axis=0)

                s, _, _ = frame_score_strict(vecR, vecU_smoothed)
                if s < 50.0:
                    id_total[tid] = max(0, id_total[tid]-1)
                elif s >= 70.0:
                    id_total[tid] = min(100, id_total[tid]+1)
            
            rank = sorted(id_total.items(), key=lambda x: x[0])
            msg = " | ".join([f"ID{tid}:{score}" for tid, score in rank])
            print(f"[total] {msg}")

        put_text(fR, "REF", (12,48), 1.1)
        for tid_r, (kps_r, conf_r, box_r) in tracks_R.items():
            draw_pose_with_id(fR, kps_r, conf_r, tid_r, box_xyxy=box_r, draw_color=(255,255,255))

        for tid_u, (kps_u, conf_u, box_u) in tracks_U.items():
            draw_pose_with_id(fU, kps_u, conf_u, tid_u, box_xyxy=box_u, draw_color=PERSON_COLORS[tid_u % len(PERSON_COLORS)])

        canvas = np.hstack([fR, fU])

        disp = canvas
        scale = args.disp_scale
        if args.disp_width is not None and args.disp_width > 0:
            scale = min(scale, float(args.disp_width) / float(canvas.shape[1]))
        if scale < 1.0:
            new_w = int(canvas.shape[1] * scale); new_h = int(canvas.shape[0] * scale)
            disp = cv2.resize(canvas, (new_w, new_h), interpolation=cv2.INTER_AREA)

        cv2.imshow("Multi Dance Compare (q to quit)", disp)
        if writer is not None: writer.write(canvas)
        if (cv2.waitKey(1) & 0xFF)==ord('q'):
            break

        frame_idx += 1
        target_fps = capR.get(cv2.CAP_PROP_FPS) or 30.0
        frame_time_ms = 1000 / target_fps
        elapsed_time_ms = (time.monotonic() - loop_start_time) * 1000
        sleep_time_ms = frame_time_ms - elapsed_time_ms
        if sleep_time_ms > 0:
            time.sleep(sleep_time_ms / 1000)

    if writer is not None: writer.release()
    capU.release(); capR.release()
    cv2.destroyAllWindows()

    if len(id_total)==0:
        print("[final] no tracked IDs.")
    else:
        ordered = sorted(id_total.items(), key=lambda x: x[0])
        print("[final] totals by ID (ascending):")
        for tid, score in ordered:
            print(f"  ID {tid}: {score}")

if __name__ == "__main__":
    main()
