import cv2
import json
import argparse
import os
from core.model_loader import load_model, make_infer
import torch

def create_json_from_video(video_path, model_path, output_json, imgsz, device, use_half, step):
    """
    Loads a video, extracts pose keypoints for each frame, and saves them to a JSON file.
    """
    model, use_half = load_model(model_path, device, use_half)
    if model is None:
        return
        
    infer_pose = make_infer(model, argparse.Namespace(
        imgsz=imgsz, device=device, conf_thres=0.25
    ), use_half)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # 비디오 속성 가져오기
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames = []
    frame_index = 0
    
    # Process frames at a given step interval
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_index % step == 0:
            print(f"Processing frame {frame_index}...")
            kps, conf = infer_pose(frame)
            
            if kps is not None:
                kps_list = kps.tolist()
            else:
                kps_list = [[float('nan'), float('nan')]] * 17
                
            if conf is not None:
                conf_list = conf.tolist()
            else:
                conf_list = [float('nan')] * 17
                
            frames.append({
                "frame_index": frame_index,
                "timestamp": cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0,
                "kps": kps_list,
                "conf": conf_list
            })
        frame_index += 1

    cap.release()
    
    # 출력 데이터 구성
    output_data = {
        "video_size": [width, height],
        "fps": fps,
        "stride": step,
        "frames": frames
    }
    
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(output_data, f, indent=4)
    
    print(f"Successfully saved {len(frames)} frames to {output_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create pose JSON from video.')
    parser.add_argument('--video_path', type=str, required=True, help='Path to the input video file.')
    parser.add_argument('--output_json', type=str, required=True, help='Path to the output JSON file.')
    parser.add_argument('--model_path', type=str, default='yolov8n-pose.pt', help='Path to the YOLO model.')
    parser.add_argument('--imgsz', type=int, default=320, help='Image size for inference.')
    parser.add_argument('--device', type=str, default=None, help='Device to use (e.g., "cpu", "cuda").')
    parser.add_argument('--step', type=int, default=1, help='Process every Nth frame.')
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    use_half = (args.device == "cuda")

    if os.path.exists(args.output_json):
        print(f"Warning: Output file '{args.output_json}' already exists. It will be overwritten.")

    create_json_from_video(
        args.video_path, args.model_path, args.output_json, args.imgsz, args.device, use_half, args.step
    )