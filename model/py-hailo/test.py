# py-hailo/test.py
import argparse
import cv2
import numpy as np
import collections
import hailo
from pose_utils import normalize_keypoints, pose_to_anglevec, frame_score_strict
from drawing_utils import draw_pose_with_id, put_text, draw_countdown_overlay
from settings import MODEL_PATH_DEFAULT, KPT_CONF_THRES, SMOOTHING_WINDOW_SIZE, PERSON_COLORS

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

def get_poses_from_tensor_output(tensor_output):
    """
    tensor_output: hailo.Tensor inference 결과
    return: {track_id: (kps, confs, bbox)}
    """
    # 여기에 HEF 모델 output parsing 로직 넣기 (SDK 제공 함수 사용)
    # 예시: hailo.get_roi_from_tensor(tensor_output)
    roi = hailo.get_roi_from_tensor(tensor_output)
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

def preprocess_reference_video(ref_video_path, hef_model, app_state):
    cap = cv2.VideoCapture(ref_video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open reference video")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        app_state.ref_frames_cache.append(frame)

        # Convert frame to tensor and run HEF inference
        input_tensor = hailo.Tensor.from_numpy(frame.astype(np.float32))
        tensor_out = hef_model.infer(input_tensor)
        poses = get_poses_from_tensor_output(tensor_out)
        app_state.ref_tracks_all_frames.append(poses)
    
    cap.release()
    print(f"[info] Preprocessing finished: {len(app_state.ref_frames_cache)} frames cached.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("--source", type=str, default="0")
    ap.add_argument("--model", type=str, default=MODEL_PATH_DEFAULT)
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--no-mirror", action="store_true")
    args = ap.parse_args()

    app_state = AppState(args)

    # 1) Load HEF model
    hef_model = hailo.HefModel(args.model)
    print("[info] HEF model loaded")

    # 2) Preprocess reference video
    preprocess_reference_video(args.ref, hef_model, app_state)

    # 3) Open user video
    cap_user = cv2.VideoCapture(args.source)
    if not cap_user.isOpened():
        raise RuntimeError("Cannot open user video")

    while True:
        ret, frame_U = cap_user.read()
        if not ret:
            break
        if not app_state.args.no_mirror:
            frame_U = cv2.flip(frame_U, 1)

        # HEF inference
        input_tensor = hailo.Tensor.from_numpy(frame_U.astype(np.float32))
        tensor_out = hef_model.infer(input_tensor)
        tracks_U = get_poses_from_tensor_output(tensor_out)

        # Reference poses for this frame
        frame_idx = app_state.frame_count
        tracks_R = app_state.ref_tracks_all_frames[frame_idx] if frame_idx < len(app_state.ref_tracks_all_frames) else {}
        frame_R = app_state.ref_frames_cache[frame_idx].copy() if frame_idx < len(app_state.ref_frames_cache) else np.zeros_like(frame_U)

        # scoring
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

        # drawing
        put_text(frame_R, "REF", (12,48), 1.1)
        for tid_r, (kps_r, conf_r, box_r) in tracks_R.items():
            draw_pose_with_id(frame_R, kps_r, conf_r, tid_r, box_xyxy=box_r, draw_color=(255,255,255))
        for tid_u, (kps_u, conf_u, box_u) in tracks_U.items():
            draw_pose_with_id(frame_U, kps_u, conf_u, tid_u, box_xyxy=box_u, draw_color=PERSON_COLORS[tid_u % len(PERSON_COLORS)])

        canvas = np.hstack([cv2.resize(frame_R, (frame_U.shape[1], frame_U.shape[0])), frame_U])
        app_state.last_rendered_frame = canvas
        app_state.increment_frame_count()

        cv2.imshow("Multi Dance Compare", canvas)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    cap_user.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

