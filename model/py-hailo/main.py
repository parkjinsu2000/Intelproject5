import argparse
import time
import collections
import numpy as np
import cv2
import os
from pathlib import Path
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import hailo

# temp
import sys
original_argv = sys.argv[:]

# Hailo-Apps imports
from hailo_apps.hailo_app_python.core.common.buffer_utils import get_numpy_from_buffer, get_caps_from_pad
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import app_callback_class
from hailo_apps.hailo_app_python.apps.pose_estimation.pose_estimation_pipeline import GStreamerPoseEstimationApp

# Project-specific imports
from settings import (
    MODEL_PATH_DEFAULT, DETECT_CONF_THRES, KPT_CONF_THRES, SMOOTHING_WINDOW_SIZE, PERSON_COLORS
)
from pose_utils import (
    normalize_keypoints, pose_to_anglevec, frame_score_strict
)
from drawing_utils import (
    draw_pose_with_id, put_text, draw_countdown_overlay
)

# --- Application State Management ---
class AppState:
    def __init__(self, args):
        self.ref_tracks_all_frames = []
        self.user_tracks_all_frames = []
        self.frame_count = 0
        self.args = args
        self.id_total = collections.defaultdict(lambda: 100)
        self.user_vec_histories = collections.defaultdict(lambda: collections.deque(maxlen=SMOOTHING_WINDOW_SIZE))
        self.last_rendered_frame = None

    def increment_frame_count(self):
        self.frame_count += 1

    def reset_frame_count(self):
        self.frame_count = 0

# --- Hailo GStreamer Callbacks ---

def get_poses_from_buffer(buffer):
    """Extracts pose information from a Hailo buffer."""
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    
    frame_poses = {}
    for detection in detections:
        if detection.get_label() != "person":
            continue

        track_id = 0
        track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
        if len(track) == 1:
            track_id = track[0].get_id()
        
        landmarks = detection.get_objects_typed(hailo.HAILO_LANDMARKS)
        if len(landmarks) > 0:
            points = landmarks[0].get_points()
            kps = np.array([[p.x(), p.y()] for p in points])
            confs = np.array([p.confidence() for p in points])
            bbox = detection.get_bbox()
            # Note: bbox format from hailo is (xmin, ymin, width, height)
            # Original code used (x1, y1, x2, y2)
            box_xyxy = np.array([bbox.xmin(), bbox.ymin(), bbox.xmin() + bbox.width(), bbox.ymin() + bbox.height()])
            
            kps[confs < KPT_CONF_THRES] = np.nan
            frame_poses[track_id] = (kps, confs, box_xyxy)
            
    return frame_poses

def ref_callback(pad, info, app_state):
    """Callback for processing the reference video."""
    buffer = info.get_buffer()
    if buffer is None: return Gst.PadProbeReturn.OK

    poses = get_poses_from_buffer(buffer)
    app_state.ref_tracks_all_frames.append(poses)
    return Gst.PadProbeReturn.OK


def user_callback(pad, info, app_state):
    """Callback for processing the user video and running the comparison logic."""
    buffer = info.get_buffer()
    if buffer is None: return Gst.PadProbeReturn.OK

    # --- Get User Frame and Poses ---
    caps = pad.get_current_caps()
    format, width, height = get_caps_from_pad(pad)
    frame_U = get_numpy_from_buffer(buffer, format, width, height)
    if app_state.args.no_mirror is False:
        frame_U = cv2.flip(frame_U, 1)
    
    tracks_U = get_poses_from_buffer(buffer)

    # --- Get Reference Frame and Poses ---
    frame_idx = app_state.frame_count
    tracks_R = {}
    if frame_idx < len(app_state.ref_tracks_all_frames):
        tracks_R = app_state.ref_tracks_all_frames[frame_idx]
    
    # Need to re-open the video to get the frame for drawing
    # This is inefficient but necessary for this app structure
    app_state.capR.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    okR, frame_R = app_state.capR.read()
    if not okR:
        # If ref video ends, use the last frame
        frame_R = np.zeros_like(frame_U)

    # --- Scoring Logic (from original main.py) ---
    vecR = None
    if tracks_R:
        try:
            ref_tid = min(tracks_R.keys())
            kps_r, _, _ = tracks_R[ref_tid]
            if np.sum(np.isfinite(kps_r)) > 10:
                kps_n_r = normalize_keypoints(kps_r)
                vecR = pose_to_anglevec(kps_n_r)
        except (ValueError, KeyError): pass

    if vecR is not None and frame_idx % max(1, app_state.args.every) == 0 and len(tracks_U) > 0:
        for tid, (kps, conf, _box) in tracks_U.items():
            if np.sum(np.isfinite(kps)) < 10: continue
            kps_n = normalize_keypoints(kps)
            vecU = pose_to_anglevec(kps_n)
            app_state.user_vec_histories[tid].append(vecU)
            vecU_smoothed = np.mean(list(app_state.user_vec_histories[tid]), axis=0)
            s, _, _ = frame_score_strict(vecR, vecU_smoothed)
            if s < 50.0: app_state.id_total[tid] = max(0, app_state.id_total[tid]-1)
            elif s >= 70.0: app_state.id_total[tid] = min(100, app_state.id_total[tid]+1)
        
        rank = sorted(app_state.id_total.items(), key=lambda x: x[0])
        msg = " | ".join([f"ID{tid}:{score}" for tid, score in rank])
        print(f"[total] {msg}")

    # --- Drawing Logic ---
    put_text(frame_R, "REF", (12,48), 1.1)
    for tid_r, (kps_r, conf_r, box_r) in tracks_R.items():
        draw_pose_with_id(frame_R, kps_r, conf_r, tid_r, box_xyxy=box_r, draw_color=(255,255,255))

    for tid_u, (kps_u, conf_u, box_u) in tracks_U.items():
        draw_pose_with_id(frame_U, kps_u, conf_u, tid_u, box_xyxy=box_u, draw_color=PERSON_COLORS[tid_u % len(PERSON_COLORS)])

    canvas = np.hstack([cv2.resize(frame_R, (frame_U.shape[1], frame_U.shape[0])), frame_U])
    app_state.last_rendered_frame = canvas

    app_state.increment_frame_count()
    return Gst.PadProbeReturn.OK

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("--source",type=str,default="0")
    ap.add_argument("--start",type=float,default=3.0)
    ap.add_argument("--every",type=int,default=5)
    ap.add_argument("--model",type=str,default=MODEL_PATH_DEFAULT, help="Path to the HEF model file.")
    ap.add_argument("--disp-scale",type=float,default=1.0)
    ap.add_argument("--disp-width",type=int,default=None)
    ap.add_argument("--save",type=str,default=None)
    ap.add_argument("--no-mirror",action="store_true")
    args=ap.parse_args()

    # --- Setup ---
    app_state = AppState(args)
    cv2.setUseOptimized(True); cv2.setNumThreads(0)

    # --- 1. Pre-process Reference Video ---
    # ===============================
    # TODO: Reference preprocessing cache
    # - Preprocessed keypoints are saved into npz
    # - If npz exists for the same reference video, load it instead of recomputing
    # - If user forces re-preprocessing, overwrite npz
    # ===============================

    # ===============================
    # TODO: Input handling restoration
    # - Restore original input argument parsing (video file vs webcam index)
    # - args.source should determine input type
    # - Keep preprocessing independent from input (preprocessing uses reference video only)
    # ===============================

    print("[info] Pre-processing reference video...")

    cap_ref = cv2.VideoCapture(args.ref)
    if not cap_ref.isOpened():
        raise RuntimeError(f"Cannot open reference video: {args.ref}")

    frame_idx = 0
    while True:
        ok, frame = cap_ref.read()
        if not ok:
            break

        poses = {}
        app_state.ref_tracks_all_frames.append(poses)

        frame_idx += 1

    cap_ref.release()
    print(f"[info] Pre-processing finished. {len(app_state.ref_tracks_all_frames)} frames processed.")

    # --- 2. Setup for User Video ---
    try: cam_idx=int(args.source); src_is_cam=True
    except ValueError: cam_idx=None; src_is_cam=False

    app_state.capR = cv2.VideoCapture(args.ref)
    if not app_state.capR.isOpened(): raise RuntimeError(f"Cannot open reference video for drawing: {args.ref}")
    W = int(app_state.capR.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(app_state.capR.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, int(round(app_state.capR.get(cv2.CAP_PROP_FPS) or 30.0)), (W*2, H))

    # --- 3. Countdown ---
    if src_is_cam:
        # This part remains largely the same, using cv2 to capture for the countdown
        capU_countdown = cv2.VideoCapture(cam_idx)
        okR0, ref_first = app_state.capR.read()
        if not okR0: raise RuntimeError("Reference has no frames.")
        ref_first = cv2.resize(ref_first, (W, H))

        t0 = time.monotonic()
        countdown = max(0.0, float(args.start))
        print(f"[info] countdown {countdown:.1f}s ...")
        while True:
            now = time.monotonic()
            remain = countdown - (now - t0)
            if remain <= -0.2: break
            okU, fU = capU_countdown.read()
            if not okU: continue
            fU = cv2.resize(fU, (W,H))
            left = ref_first.copy()
            right = fU.copy()
            text = str(int(np.ceil(remain))) if remain > 0.8 else "START"
            draw_countdown_overlay(left, text); draw_countdown_overlay(right, text)
            put_text(left, "REF", (12,48), 1.1); put_text(right, f"USER [{'mirror' if not args.no_mirror else 'normal'}]", (12,48), 1.0)
            canvas = np.hstack([left, right])
            cv2.imshow("Multi Dance Compare (q to quit)", canvas)
            if (cv2.waitKey(1) & 0xFF)==ord('q'):
                if writer is not None: writer.release()
                capU_countdown.release(); app_state.capR.release(); cv2.destroyAllWindows(); return
        capU_countdown.release()

    # --- 4. Run Main Application ---
    print("[info] scoring start!")
    app_state.reset_frame_count()
    sys.argv = [original_argv[0], '--input', args.source] # Set input for the user app
    user_app = GStreamerPoseEstimationApp(user_callback, app_state)
    sys.argv = original_argv # Restore original arguments

    # The user_app.run() will block, so we need a separate thread/loop to display frames
    # For simplicity, we'll rely on the pipeline being fast and just show the last frame
    # A more robust solution would use a separate display thread.
    main_loop = GLib.MainLoop()
    def on_eos(bus, msg, loop):
        loop.quit()

    user_app.pipeline.bus.add_signal_watch()
    user_app.pipeline.bus.connect("message::eos", on_eos, main_loop)

    try:
        while not main_loop.is_running():
            if app_state.last_rendered_frame is not None:
                disp = app_state.last_rendered_frame
                # Display scaling logic from original code
                scale = args.disp_scale
                if args.disp_width is not None and args.disp_width > 0:
                    scale = min(scale, float(args.disp_width) / float(disp.shape[1]))
                if scale < 1.0:
                    new_w = int(disp.shape[1] * scale); new_h = int(disp.shape[0] * scale)
                    disp = cv2.resize(disp, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
                cv2.imshow("Multi Dance Compare (q to quit)", disp)
                if writer is not None:
                    writer.write(app_state.last_rendered_frame)

            if (cv2.waitKey(1) & 0xFF)==ord('q'):
                user_app.pipeline.send_event(Gst.Event.new_eos())
                main_loop.quit()
                break
            
            # A small sleep to prevent a busy-wait loop spinning at 100% CPU
            time.sleep(0.01)

    except KeyboardInterrupt:
        user_app.pipeline.send_event(Gst.Event.new_eos())
        main_loop.quit()

    # --- 5. Final Score ---
    if writer is not None: writer.release()
    app_state.capR.release()
    cv2.destroyAllWindows()

    if len(app_state.id_total)==0:
        print("[final] no tracked IDs.")
    else:
        ordered = sorted(app_state.id_total.items(), key=lambda x: x[0])
        print("[final] totals by ID (ascending):")
        for tid, score in ordered:
            print(f"  ID {tid}: {score}")

if __name__ == "__main__":
    # Set up GStreamer environment from hailo_rpi5_examples
    if "HAILO_VIRTUAL_ENV" in os.environ:
        project_root = Path(os.environ["HAILO_VIRTUAL_ENV"]).parent
        os.environ["HAILO_ENV_FILE"] = str(project_root / ".env")
    main()
