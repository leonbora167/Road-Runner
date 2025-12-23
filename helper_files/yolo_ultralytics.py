import cv2 
import numpy as np
import logging, os
 
log_file = "logs/yolo_np.log"
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    handlers=[
                        logging.FileHandler(log_file, mode="a")
                        ])

def yolo_postprocess_vehicle(results):
    if len(results[0].boxes) == 0:
        return [] # No detections
    else:
        n_det = len(results[0].boxes.cls)
        i = results[0]
        response = []
        for index in range(n_det):
            boxes_xyxy = i.boxes.xyxy
            x1,y1,x2,y2 = map(int, boxes_xyxy.detach().cpu().numpy()[index])
            class_id = i.boxes.cls[index].cpu().numpy().item()
            if class_id == 2 or class_id == 5 or class_id == 7:
                confidence = i.boxes.conf[index].cpu().numpy().item()
                lis = [x1,y1,x2,y2,class_id,confidence]
                response.append(lis)
        if(response == []):
            return [] # No valid det class
        else:
            return response
        
def yolo_postprocess_np(results, vehicle_image):
    if len(results[0].boxes) == 0:
        logging.info("No number plate found in the image")
        return [] # No dets 
    else:
        i = results[0] 
        for index in range(1):
            boxes_xyxy = i.boxes.xyxy
            #print(boxes_xyxy)
            x1,y1,x2,y2 = map(int, boxes_xyxy.detach().cpu().numpy()[index])
            class_id = i.boxes.cls[index].cpu().numpy().item()
            confidence = i.boxes.conf[index].cpu().numpy().item()
            response = [x1,y1,x2,y2, class_id, confidence]
            np_image = vehicle_image[y1:y2, x1:x2]
        return response, np_image
