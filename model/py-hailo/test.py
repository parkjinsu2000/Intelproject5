from hailo_apps.hailo_app_python.apps.pose_estimation.pose_estimation_pipeline import GStreamerPoseEstimationApp
from hailo_apps.hailo_app_python.core.common.buffer_utils import get_numpy_from_buffer, get_caps_from_pad

class MyPoseApp(GStreamerPoseEstimationApp):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # appsink에서 callback 오버라이드
    def on_frame(self, sample):
        buffer = sample.get_buffer()
        caps = sample.get_caps()
        numpy_frame = get_numpy_from_buffer(buffer, caps)
        # numpy_frame은 HxWxC 형태로 이미지 데이터
        poses = self.extract_pose(numpy_frame)
        self.pose_callback(poses)

    def extract_pose(self, frame):
        # 모델 결과를 frame에서 뽑는 로직
        # centerpose 출력 tensor → keypoints 변환
        # 예시: [N x 17 x (x, y, score)]
        return frame['keypoints']  # 실제 구조 확인 필요

    def pose_callback(self, poses):
        # 여기서 원하는 처리를 수행
        print("Frame poses:", poses)

app = MyPoseApp(input_source="ref.mp4")
app.run()
