import numpy as np

def get_person_center(kps):
    """
    keypoints 배열에서 사람의 중심 좌표(cx, cy)를 계산합니다.
    기본적으로 어깨(5,6)와 엉덩이(11,12) 평균을 사용합니다.
    """
    torso_indices = [5, 6, 11, 12]
    valid_pts = [kps[i] for i in torso_indices if np.all(np.isfinite(kps[i]))]
    if len(valid_pts) == 0:
        return None
    return np.mean(valid_pts, axis=0)  # (cx, cy)

def classify_region(cx, frame_width):
    """
    화면을 좌/중/우로 3등분하여 a/s/d 반환
    """
    one_third = frame_width / 3
    if cx < one_third:
        return 'd'
    elif cx < 2 * one_third:
        return 's'
    else:
        return 'a'
