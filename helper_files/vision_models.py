import numpy as np 
import cv2 

def tracks_xyxy(tracks):
    lis = []
    for t in tracks:
        x1, y1, x2, y2 = map(int, t[:4])
        track_id       = int(t[4])
        conf           = float(t[5])
        class_id       = int(t[6])
        lis.append([x1,y1,x2,y2, conf, class_id, track_id])
    return lis

def yolo_to_boxmot(yolo_outputs):
    if len(yolo_outputs) == 0:
        return np.empty((0, 6)) #Ultralytics yolo has 6 outputs per image as x1,y1,x2,y2,conf,class
    
    dets = []
    for x1, y1, x2, y2, cls, conf in yolo_outputs:
        dets.append([x1, y1, x2, y2, conf, cls])
    
    return np.array(dets, dtype=np.float32)