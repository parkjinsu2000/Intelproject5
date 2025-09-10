
import cv2
import numpy as np
from settings import EDGES, KPT_CONF_THRES

def draw_pose_with_id(img,kps_xy,kps_conf,tid:int,box_xyxy=None,conf_thres=KPT_CONF_THRES,draw_color=(0,255,0)):
    if kps_xy is None: return
    kps_xy=np.array(kps_xy,dtype=np.float32)
    kpt_radius=2; line_thickness=1
    for i,j in EDGES:
        if i<17 and j<17:
            pi,pj = kps_xy[i],kps_xy[j]
            if np.all(np.isfinite(pi)) and np.all(np.isfinite(pj)):
                cv2.line(img,tuple(np.round(pi).astype(int)),tuple(np.round(pj).astype(int)),draw_color,line_thickness,cv2.LINE_AA)
    for idx,p in enumerate(kps_xy):
        if not np.all(np.isfinite(p)): continue
        color = draw_color
        if kps_conf is not None:
            c = kps_conf[idx]
            if (not np.isfinite(c)) or (c<conf_thres):
                color = (0,0,255)
        cv2.circle(img,tuple(np.round(p).astype(int)),kpt_radius,color,-1,cv2.LINE_AA)
    if box_xyxy is not None:
        x1,y1,x2,y2=box_xyxy
        lx,ly=int(x1),int(max(10,y1-8))
    else:
        finite=kps_xy[np.isfinite(kps_xy).all(axis=1)]
        if len(finite):
            x_min=int(np.min(finite[:,0]))
            y_min=int(np.min(finite[:,1]))
            lx,ly=x_min,max(10,y_min-8)
        else:
            lx,ly=12,28
    label=f"ID {tid}"
    cv2.putText(img,label,(lx,ly),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,0),3,cv2.LINE_AA)
    cv2.putText(img,label,(lx,ly),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2,cv2.LINE_AA)

def put_text(img,text,org=(12,48),scale=1.0,color=(255,255,255)):
    cv2.putText(img,text,org,cv2.FONT_HERSHEY_SIMPLEX,scale,(0,0,0),3,cv2.LINE_AA)
    cv2.putText(img,text,org,cv2.FONT_HERSHEY_SIMPLEX,scale,color,2,cv2.LINE_AA)

def draw_countdown_overlay(img,ttext):
    H,W=img.shape[:2]; scale=max(1.0,min(H,W)/180.0); thickness=max(2,int(scale*4))
    (tw,th),_=cv2.getTextSize(ttext,cv2.FONT_HERSHEY_SIMPLEX,scale,thickness)
    x=(W-tw)//2; y=(H+th)//2
    cv2.putText(img,ttext,(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,(0,0,0),thickness+2,cv2.LINE_AA)
    cv2.putText(img,ttext,(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,(255,255,255),thickness,cv2.LINE_AA)
