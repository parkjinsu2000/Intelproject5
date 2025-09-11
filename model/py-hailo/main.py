# py-hailo/main.py
import argparse
import time
import collections
import numpy as np
import cv2
import os
from pathlib import Path
import gi
import threading
import sys

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import hailo

# hailo-apps helpers (from your environment)
from hailo_apps.hailo_app_python.core.common.buffer_utils import get_numpy_from_buffer, get_caps_from_pad
from hailo_apps.hailo_app_python.apps.pose_estimation.pose_estimation_pipeline import GStreamerPoseEstimationApp

from settings import MODEL_PATH_DEFAULT, KPT_CONF_THRES, SMOOTHING_WINDOW_SIZE, PERSON_COLORS
from pose_utils import normalize_keypoints, pose_to_anglevec, frame_score_strict
from drawing_utils import draw_pose_with_id, put_text, draw_countdown_overlay

class AppState:
    def __init__(self, args):
        self.ref_tracks_all_frames = []
        self.ref_frames_cache = []
        self.user_vec_histories = collections.defaultdict(lambda: collections.deque(maxlen=SMOOTHING_WINDOW_SIZE))
        self.last_rendered_frame = None
        self.frame_count = 0
        self.args = args
        self.id_total = collections.defaultdict(lambda: 100)
        self.running = True

    def increment_frame_count(self):
        self.frame_count += 1

def get_poses_from_buffer(buffer):
    # Same as original: extract hailo roi from buffer
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
            box_xyxy = np.array([bbox.xmin(), bbox.ymin(), bbox.xmin() + bbox.width(), bbox.ymin() + bbox.height()])
            kps[confs < KPT_CONF_THRES] = np.nan
            frame_poses[track_id] = (kps, confs, box_xyxy)
    return frame_poses

def ref_callback(pad, info, app_state):
    buffer = info.get_buffer()
    if buffer is None: return Gst.PadProbeReturn.OK
    poses = get_poses_from_buffer(buffer)
    app_state.ref_tracks_all_frames.append(poses)
    return Gst.PadProbeReturn.OK

def user_callback(pad, info, app_state):
    if not app_state.running:
        return Gst.PadProbeReturn.DROP
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    # decode frame for display
    format, width, height = get_caps_from_pad(pad)
    frame_U = get_numpy_from_buffer(buffer, format, width, height)
    if not app_state.args.no_mirror:
        frame_U = cv2.flip(frame_U, 1)

    # get poses
    tracks_U = get_poses_from_buffer(buffer)

    # get reference poses for this frame index
    frame_idx = app_state.frame_count
    tracks_R = app_state.ref_tracks_all_frames[frame_idx] if frame_idx < len(app_state.ref_tracks_all_frames) else {}
    # use cached ref frame for drawing
    if frame_idx < len(app_state.ref_frames_cache):
        frame_R = app_state.ref_frames_cache[frame_idx].copy()
    else:
        frame_R = np.zeros_like(frame_U)

    # scoring (same logic as before)
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

    if frame_idx % 60 == 0:
        print(f"[total] {dict(app_state.id_total)}", flush=True)

    # drawing
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
    ap.add_argument("--model",type=str,default=MODEL_PATH_DEFAULT)
    ap.add_argument("--disp-scale",type=float,default=1.0)
    ap.add_argument("--disp-width",type=int,default=None)
    ap.add_argument("--save",type=str,default=None)
    ap.add_argument("--no-mirror",action="store_true")
    args=ap.parse_args()

    app_state = AppState(args)
    cv2.setUseOptimized(True); cv2.setNumThreads(4)

    # 1) cache reference frames (for drawing)
    print("[info] Pre-processing reference video and caching frames...")
    cap_ref = cv2.VideoCapture(args.ref)
    if not cap_ref.isOpened(): raise RuntimeError("Cannot open reference video")
    while True:
        ok, frame = cap_ref.read()
        if not ok: break
        app_state.ref_frames_cache.append(frame)
        # keep ref_tracks_all_frames empty for now; the ref pipeline run below (GStreamer) will fill poses
        app_state.ref_tracks_all_frames.append({})
    cap_ref.release()
    print(f"[info] Pre-processing finished. {len(app_state.ref_frames_cache)} frames cached.")

    # 2) run Hailo GStreamer reference pass to fill ref_tracks_all_frames
    # Use the provided GStreamerPoseEstimationApp to process the reference video.
    original_argv = sys.argv[:]
    sys.argv = [original_argv[0], '--input', args.ref]
    ref_app = GStreamerPoseEstimationApp(ref_callback, app_state)
    # This will run and call ref_callback for each frame, filling app_state.ref_tracks_all_frames
    ref_app.run()
    print("[info] Reference preprocessing done. Poses extracted.")

    # 3) set up user pipeline (inference) but disable display on the pipeline (use fakesink)
    sys.argv = [original_argv[0], '--input', args.source]
    user_app = GStreamerPoseEstimationApp(user_callback, app_state)
    sys.argv = original_argv

    # change hailo_display to fakesink to avoid X display issues
    sink_elem = user_app.pipeline.get_by_name("hailo_display")
    if sink_elem is not None:
        sink_elem.set_property("video-sink", Gst.ElementFactory.make("fakesink", "fakesink"))
        print("[info] Set hailo_display -> fakesink (no pipeline-level display)")

    # run GLib mainloop in separate thread
    main_loop = GLib.MainLoop()
    def gst_thread_func():
        main_loop.run()
        app_state.running = False
    gst_thread = threading.Thread(target=gst_thread_func)
    gst_thread.start()
    print("[info] GStreamer thread started.")

    writer = None
    if args.save:
        # optionally save frames (implement writer earlier if needed)
        h = app_state.ref_frames_cache[0].shape[0]
        w = app_state.ref_frames_cache[0].shape[1] * 2
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, 25.0, (w, h))


    # main display loop - show last_rendered_frame via OpenCV
    try:
        while app_state.running:
#            if app_state.last_rendered_frame is not None:
            if app_state.last_rendered_frame is not None and (app_state.frame_count % DISPLAY_EVERY == 0):
                disp = app_state.last_rendered_frame
                cv2.imshow("Multi Dance Compare (q to quit)", disp)
                if writer is not None and app_state.last_rendered_frame is not None:
                    writer.write(app_state.last_rendered_frame)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                user_app.pipeline.send_event(Gst.Event.new_eos())
                break
            time.sleep(0.01)
    except KeyboardInterrupt:
        user_app.pipeline.send_event(Gst.Event.new_eos())
    finally:
        if main_loop.is_running(): main_loop.quit()
        gst_thread.join(timeout=5)
        cv2.destroyAllWindows()
        print("[info] Finished.")
        if app_state.id_total:
            for tid, score in sorted(app_state.id_total.items()): print(f"ID {tid}: {score}")
        if writer is not None:
            writer.release()

if __name__ == "__main__":
    main()

