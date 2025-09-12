# py-hailo/main.py
import argparse
import os
import time
import collections
import numpy as np
import cv2
import threading
import sys
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import hailo

from hailo_apps.hailo_app_python.core.common.buffer_utils import get_numpy_from_buffer, get_caps_from_pad
from hailo_apps.hailo_app_python.apps.pose_estimation.pose_estimation_pipeline import GStreamerPoseEstimationApp

from settings import MODEL_PATH_DEFAULT, KPT_CONF_THRES, SMOOTHING_WINDOW_SIZE, PERSON_COLORS
from pose_utils import normalize_keypoints, pose_to_anglevec, frame_score_strict
from drawing_utils import draw_pose_with_id, put_text, draw_countdown_overlay

class AppState:
    def __init__(self, args):
        self.ref_tracks_all_frames = []
        self.ref_total_frames = 0
        self.user_vec_histories = collections.defaultdict(lambda: collections.deque(maxlen=SMOOTHING_WINDOW_SIZE))
        self.last_rendered_frame = None
        self.frame_count = 0
        self.args = args
        self.id_total = collections.defaultdict(lambda: 100)
        self.running = True

    def increment_frame_count(self):
        self.frame_count += 1

def get_poses_from_buffer(buffer):
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
    if buffer is None:
        return Gst.PadProbeReturn.OK
    poses = get_poses_from_buffer(buffer)
    app_state.ref_tracks_all_frames.append(poses)

    # Progress bar
    total_frames = app_state.ref_total_frames
    if total_frames > 0:
        current_frame = len(app_state.ref_tracks_all_frames)
        progress = (current_frame / total_frames) * 100
        # Use a carriage return to show progress on a single line
        print(f"\r> Preprocessing reference: [{current_frame}/{total_frames}] {progress:.1f}%", end="", flush=True)

    return Gst.PadProbeReturn.OK

def user_callback(pad, info, app_state):
    if not app_state.running:
        return Gst.PadProbeReturn.DROP

    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    # Get user frame and poses
    format, width, height = get_caps_from_pad(pad)
    frame_U = get_numpy_from_buffer(buffer, format, width, height)
    if not app_state.args.no_mirror:
        frame_U = cv2.flip(frame_U, 1)
    tracks_U = get_poses_from_buffer(buffer)

    # Get reference poses for scoring
    frame_idx = app_state.frame_count
    tracks_R = app_state.ref_tracks_all_frames[frame_idx] if frame_idx < len(app_state.ref_tracks_all_frames) else {}

    # Scoring logic (remains the same)
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
            if np.sum(np.isfinite(kps)) < 10:
                continue
            kps_n = normalize_keypoints(kps)
            vecU = pose_to_anglevec(kps_n)
            app_state.user_vec_histories[tid].append(vecU)
            vecU_smoothed = np.mean(list(app_state.user_vec_histories[tid]), axis=0)
            s, _, _ = frame_score_strict(vecR, vecU_smoothed)
            if s < 50.0: 
                app_state.id_total[tid] = max(0, app_state.id_total[tid]-1)
            elif s >= 70.0:
                app_state.id_total[tid] = min(100, app_state.id_total[tid]+1)

    # Simplified drawing: only draw user poses on user frame
    for tid_u, (kps_u, conf_u, box_u) in tracks_U.items():
        draw_pose_with_id(frame_U, kps_u, conf_u, tid_u, box_xyxy=box_u, draw_color=PERSON_COLORS[tid_u % len(PERSON_COLORS)])

    # Update app state for display
    app_state.last_rendered_frame = frame_U
    app_state.last_score = dict(app_state.id_total)

    # Print scores
    scores = dict(app_state.id_total)
    score_str = " | ".join([f"ID{tid}: {score}" for tid, score in sorted(scores.items())])
    print(f"[total] {score_str}", flush=True)

    app_state.increment_frame_count()
    return Gst.PadProbeReturn.OK

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("--source", type=str, default="0")
    ap.add_argument("--start", type=float, default=3.0)
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--model", type=str, default=MODEL_PATH_DEFAULT)
    ap.add_argument("--disp-scale", type=float, default=1.0)
    ap.add_argument("--disp-width", type=int, default=None)
    ap.add_argument("--save", type=str, default=None)
    ap.add_argument("--no-mirror", action="store_true")
    
    # Store our script's args and clear sys.argv for the Hailo apps
    my_script_args = ap.parse_args()
    original_sys_argv = sys.argv[:]
    sys.argv = [original_sys_argv[0]]

    app_state = AppState(my_script_args)
    cv2.setUseOptimized(True)
    cv2.setNumThreads(4)

    # 1) Get reference video frame count
    print("[info] Getting reference video frame count...")
    cap_ref = cv2.VideoCapture(my_script_args.ref)
    if not cap_ref.isOpened():
        raise RuntimeError("Cannot open reference video")
    app_state.ref_total_frames = int(cap_ref.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_ref.release()
    print(f"[info] Reference video has {app_state.ref_total_frames} total frames.")

    # 2) Hailo GStreamer reference pass
    npz_path = f"{my_script_args.ref}.npz"
    if os.path.exists(npz_path):
        print(f"[info] Loading pre-processed poses from {npz_path}")
        with np.load(npz_path, allow_pickle=True) as data:
            app_state.ref_tracks_all_frames = data['poses']
        print(f"[info] {len(app_state.ref_tracks_all_frames)} frames of poses loaded.")
    else:
        print("[info] Reference preprocess pose extraction start (will save to .npz).")
        
        # Set argv specifically for the reference app
        sys.argv.extend(['--input', os.path.abspath(my_script_args.ref)])
        
        # Create the app, but we will run our own main loop
        ref_app = GStreamerPoseEstimationApp(ref_callback, app_state)

        # --- Start of manual probe injection ---
        identity_element = ref_app.pipeline.get_by_name("identity_callback")
        if identity_element:
            sink_pad = identity_element.get_static_pad("sink")
            if sink_pad:
                probe_id = sink_pad.add_probe(Gst.PadProbeType.BUFFER, ref_callback, app_state)
                if probe_id > 0:
                    print(f"[info] Manually added probe with ID {probe_id} to identity_callback sink pad.")
                else:
                    print("[error] Failed to add probe to identity_callback sink pad.")
            else:
                print("[error] Could not get sink pad from identity_callback.")
        else:
            print("[error] Could not find element named identity_callback in the pipeline.")
        # --- End of manual probe injection ---

        # We only need the data from the callback, not the video display.
        # So we get the display element and replace its sink with a fakesink.
        sink_elem = ref_app.pipeline.get_by_name("hailo_display")
        if sink_elem is not None:
            # sink_elem.set_property("video-sink", Gst.ElementFactory.make("fakesink", "fakesink"))
            print("[info] fakesink를 비활성화하고, 비디오 출력을 활성화합니다.")

        # Create and run our own main loop to handle messages gracefully
        ref_main_loop = GLib.MainLoop()
        bus = ref_app.pipeline.get_bus()
        bus.add_signal_watch()

        def on_message(bus, message):
            mtype = message.type
            if mtype == Gst.MessageType.EOS:
                print("\n[info] EOS received, quitting preprocessing loop.")
                ref_main_loop.quit()
            elif mtype == Gst.MessageType.ERROR:
                err, debug = message.parse_error()
                print(f"\n[error] GStreamer error: {err}, {debug}")
                ref_main_loop.quit()

        bus.connect("message", on_message)

        # Start the pipeline and the main loop
        print("[info] Reference pipeline starting...")
        ref_app.pipeline.set_state(Gst.State.PLAYING)
        ref_main_loop.run() # This blocks until quit() is called

        # Cleanup
        ref_app.pipeline.set_state(Gst.State.NULL)
        print(f"[info] Reference preprocessing done. Saving {len(app_state.ref_tracks_all_frames)} frames of poses to {npz_path}")
        np.savez(npz_path, poses=app_state.ref_tracks_all_frames)
        
        # Clear argv for the next app
        sys.argv = [original_sys_argv[0]]

    # 3) user pipeline
    # Set argv specifically for the user app
    sys.argv.extend(['--input', my_script_args.source])
    user_app = GStreamerPoseEstimationApp(user_callback, app_state)
    sys.argv = original_sys_argv # Restore original argv

    # --- Start of manual probe injection ---
    identity_element_user = user_app.pipeline.get_by_name("identity_callback")
    if identity_element_user:
        sink_pad_user = identity_element_user.get_static_pad("sink")
        if sink_pad_user:
            probe_id_user = sink_pad_user.add_probe(Gst.PadProbeType.BUFFER, user_callback, app_state)
            if probe_id_user > 0:
                print(f"[info] Manually added probe with ID {probe_id_user} to user pipeline identity_callback sink pad.")
        else:
            print("[error] Could not get sink pad from user pipeline identity_callback.")
    else:
        print("[error] Could not find element named identity_callback in the user pipeline.")
    # --- End of manual probe injection ---

    # Set up a bus watcher to handle messages from the pipeline
    bus = user_app.pipeline.get_bus()
    bus.add_signal_watch()

    def on_user_message(bus, message):
        mtype = message.type
        if mtype == Gst.MessageType.EOS:
            print(f"\n[info] Received {Gst.MessageType.get_name(mtype)}, quitting user loop.")
            main_loop.quit()
        elif mtype == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"\n[error] GStreamer error: {err}, {debug}")
            main_loop.quit()

    bus.connect("message", on_user_message)

    sink_elem = user_app.pipeline.get_by_name("hailo_display")
    if sink_elem is not None:
        #sink_elem.set_property("video-sink", Gst.ElementFactory.make("fakesink", "fakesink"))
        print("[info] user_app: fakesink를 비활성화하고, 비디오 출력을 활성화합니다.")

    main_loop = GLib.MainLoop()
    def gst_thread_func():
        main_loop.run()
        app_state.running = False
    gst_thread = threading.Thread(target=gst_thread_func)

    print("[info] User pipeline starting...")
    user_app.pipeline.set_state(Gst.State.PLAYING)

    gst_thread.start()
    print("[info] GStreamer thread started.")

    # writer = None
    # if my_script_args.save:
    #     h, w = app_state.ref_frames_cache[0].shape[:2]
    #     w *= 2  # canvas width
    #     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    #     writer = cv2.VideoWriter(my_script_args.save, fourcc, 25.0, (w, h))

    WINDOW_NAME = "Multi Dance Compare (q to quit)" # Define WINDOW_NAME outside try block
    cv2.namedWindow(WINDOW_NAME, cv2_WINDOW_NORMAL)

    try:
        while app_state.running:
            if app_state.last_rendered_frame is not None:
                cv2.imshow(WINDOW_NAME, app_state.last_rendered_frame) # Use WINDOW_NAME here

            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                print("\n'q' key pressed. Shutting down.")
                app_state.running = False
                main_loop.quit()
                break

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nCtrl+C pressed. Shutting down.")
        app_state.running = False
        main_loop.quit()

    finally:
        if main_loop.is_running():
            main_loop.quit()
        gst_thread.join(timeout=5)
        cv2.destroyAllWindows()
        for tid, score in sorted(app_state.id_total.items()):
            print(f"ID {tid}: {score}")
        # if writer is not None:
        #     writer.release()
        user_app.pipeline.set_state(Gst.State.NULL)
        print("[info] Finished.")

if __name__ == "__main__":
    main()

