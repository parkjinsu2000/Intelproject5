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

<<<<<<< HEAD
=======
    # 비디오 속성 가져오기
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

>>>>>>> 232d96d606b13b5b5a38cea1d7d1258e10a80353
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
            
<<<<<<< HEAD
            # Convert numpy arrays to lists for JSON serialization
            if kps is not None:
                kps_list = kps.tolist()
            else:
                # kps_list = None
=======
            if kps is not None:
                kps_list = kps.tolist()
            else:
>>>>>>> 232d96d606b13b5b5a38cea1d7d1258e10a80353
                kps_list = [[float('nan'), float('nan')]] * 17
                
            if conf is not None:
                conf_list = conf.tolist()
            else:
<<<<<<< HEAD
                # conf_list = None
=======
>>>>>>> 232d96d606b13b5b5a38cea1d7d1258e10a80353
                conf_list = [float('nan')] * 17
                
            frames.append({
                "frame_index": frame_index,
                "timestamp": cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0,
                "kps": kps_list,
                "conf": conf_list
            })
        frame_index += 1

    cap.release()
    
<<<<<<< HEAD
    os.makedirs(os.path.dirname(output_json), exist_ok=True)  # 디렉터리 자동 생성
    with open(output_json, 'w') as f:
        json.dump({"frames": frames}, f, indent=4)
=======
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
>>>>>>> 232d96d606b13b5b5a38cea1d7d1258e10a80353
    
    print(f"Successfully saved {len(frames)} frames to {output_json}")

if __name__ == "__main__":
<<<<<<< HEAD
    # 💡 You can set the video and output paths directly here
    video_dir = "resources/videos"
    video_filename = "frog.mp4" # 👈 여기에 동영상 파일 이름을 입력하세요
    video_path = os.path.join(video_dir, video_filename)
    
    output_json = os.path.join(video_dir, "frog.json") # 👈 여기에 출력할 JSON 파일 이름을 입력하세요
    
    # Optional arguments, you can change them as needed
    model_path = "yolov8n-pose.pt"
    imgsz = 320
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_half = False
    step = 10
    
    if not output_json.endswith('.json'):
        output_json += '.json'
    
    if os.path.exists(output_json):
        print(f"Warning: Output file '{output_json}' already exists. It will be overwritten.")
        
    create_json_from_video(
        video_path, model_path, output_json, imgsz, device, use_half, step
=======
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
>>>>>>> 232d96d606b13b5b5a38cea1d7d1258e10a80353
    )