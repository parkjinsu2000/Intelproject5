
import numpy as np

# -------------------- 설정 --------------------
MODEL_PATH_DEFAULT = "yolov8n-pose.pt"
DETECT_CONF_THRES = 0.25
KPT_CONF_THRES = 0.20
K_STRICT = 10.0
MARGIN = 0.02
SMOOTHING_WINDOW_SIZE = 15

PERSON_COLORS = [
    (0, 255, 0),(0, 0, 255),(255, 0, 0),(0, 255, 255),(255, 0, 255),
    (255, 255, 0),(128, 0, 128),(0, 128, 128),(128, 128, 0),(0, 0, 128)
]

# COCO 17 keypoints
NOSE=0; L_EYE=1; R_EYE=2; L_EAR=3; R_EAR=4
L_SH=5; R_SH=6; L_EL=7; R_EL=8
L_WR=9; R_WR=10; L_HP=11; R_HP=12
L_KN=13; R_KN=14; L_AN=15; R_AN=16

ANGLE_TRIPLES = [
    (L_SH,L_EL,L_WR),(R_SH,R_EL,R_WR),
    (L_HP,L_KN,L_AN),(R_HP,R_KN,R_AN),
    (L_SH,L_HP,L_KN),(R_SH,R_HP,R_KN)
]

EDGES = [
    (5,7),(7,9),(6,8),(8,10),
    (11,13),(13,15),(12,14),(14,16),
    (5,6),(11,12),(5,11),(6,12),
    (0,5),(0,6),(0,1),(0,2),(1,3),(2,4)
]
