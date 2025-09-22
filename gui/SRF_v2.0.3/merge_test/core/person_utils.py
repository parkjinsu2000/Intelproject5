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

def get_midpoint_between_people(kps_list):
    """
    여러 사람의 keypoints 리스트에서
    첫 두 사람의 중심을 구해 그 중간 좌표를 반환합니다.
    
    kps_list: [kps1, kps2, ...] 형태 (사람별 keypoints 배열)
    return: (mx, my) or None
    """
    if len(kps_list) < 2:
        return None  # 사람이 2명 이상 있어야 함

    # 각 사람 중심 좌표 구하기
    centers = []
    for kps in kps_list[:2]:  # 앞의 두 명만 사용
        center = get_person_center(kps)
        if center is not None:
            centers.append(center)

    if len(centers) < 2:
        return None

    # 두 사람 중심의 정중앙 좌표
    mx = (centers[0][0] + centers[1][0]) / 2
    my = (centers[0][1] + centers[1][1]) / 2
    return (mx, my)

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
