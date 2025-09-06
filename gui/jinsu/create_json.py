import cv2
import json
import argparse
import os
from model_loader import load_model, make_infer
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
            
            # Convert numpy arrays to lists for JSON serialization
            if kps is not None:
                kps_list = kps.tolist()
            else:
                kps_list = None
                
            if conf is not None:
                conf_list = conf.tolist()
            else:
                conf_list = None
                
            frames.append({
                "frame_index": frame_index,
                "timestamp": cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0,
                "kps": kps_list,
                "conf": conf_list
            })
        frame_index += 1

    cap.release()
    
    with open(output_json, 'w') as f:
        json.dump({"frames": frames}, f, indent=4)
    
    print(f"Successfully saved {len(frames)} frames to {output_json}")

if __name__ == "__main__":
    # 💡 You can set the video and output paths directly here
    video_dir = "videos"
    video_filename = "naruto.mp4" # 👈 여기에 동영상 파일 이름을 입력하세요
    video_path = os.path.join(video_dir, video_filename)
    
    output_json = "output.json" # 👈 여기에 출력할 JSON 파일 이름을 입력하세요
    
    # Optional arguments, you can change them as needed
    model_path = "yolov8n-pose.pt"
    imgsz = 320
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_half = False
    step = 1
    
    if not output_json.endswith('.json'):
        output_json += '.json'
    
    if os.path.exists(output_json):
        print(f"Warning: Output file '{output_json}' already exists. It will be overwritten.")
        
    create_json_from_video(
        video_path, model_path, output_json, imgsz, device, use_half, step
    )
