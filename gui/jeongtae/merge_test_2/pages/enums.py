from enum import IntEnum

class PageIndex(IntEnum):
    MAIN = 0
    RANK = 1
    VIDEO_SELECT = 2
    ADD_NEW_CHALLENGE_VIDEO = 3
    # PoseScoreApp 같은 동적 페이지는 따로 관리 (매번 새로 생성)

class ModeNumber(IntEnum):
    SINGLE = 0
    MULTIPLE = 1