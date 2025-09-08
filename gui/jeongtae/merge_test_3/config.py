import os

class DirPath:
    ROOT_DIR        = os.path.dirname(os.path.abspath(__file__))  # config.py 위치
    BASE_VIDEO_DIR  = os.path.join(ROOT_DIR, "resources", "videos")

    # 세부 비디오 디렉터리
    REF_VIDEO_DIR   = os.path.join(BASE_VIDEO_DIR, "ref")   # 기준 영상
    USER_VIDEO_DIR  = os.path.join(BASE_VIDEO_DIR, "user")  # 사용자 영상

    IMAGE_DIR       = os.path.join(ROOT_DIR, "resources", "images")
    DB_DIR          = os.path.join(ROOT_DIR, "resources", "DB")
    RANK_DIR        = os.path.join(DB_DIR, "rank")
    DETAILS_DIR     = os.path.join(DB_DIR, "details")

class FileName:
    MAIN_IMAGE      = "main_Image.png"
    RANK_FILE       = "rank_video_list.txt"