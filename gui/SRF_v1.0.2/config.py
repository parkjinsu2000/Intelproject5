import os

# -------------------------------
# 페이지 인덱스 (Control / View)
# -------------------------------
class ControlPageIndex:
    MAIN                    = 0
    VIDEO_SELECT            = 1
    BACKGROUND_VIDEO_PLAY   = 2
    SCORE                   = 3
    AVATAR_SELECT           = 4
    AVATAR_LOADING          = 5
    CONVERTED_AVATAR_VIDEO  = 6


class ViewPageIndex:
    MAIN                    = 0
    VIDEO_SELECT            = 1
    GAME_PLAY               = 2
    SCORE                   = 3
    AVATAR_SELECT           = 4
    AVATAR_LOADING          = 5
    CONVERTED_AVATAR_VIDEO  = 6


# -------------------------------
# 리소스 경로
# -------------------------------
class SourcePath:
    # config.py 파일이 있는 디렉토리 절대 경로
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    BG_PATH = ""   # 필요시 여기에 배경 동영상 경로 지정
    THUMBNAIL_BTN = os.path.join(BASE_DIR, "resource/button1.png")


# -------------------------------
# 모니터 인덱스 설정
# -------------------------------
class MonitorIndex:
    # ⚠️ 모니터 인덱스는 OS/환경마다 달라질 수 있음
    # QGuiApplication.screens() 로 가져온 순서대로 0,1,2... 로 번호가 매겨짐
    VIEW = 0       # View 패널이 표시될 모니터
    CONTROL = 1    # Control 패널이 표시될 모니터