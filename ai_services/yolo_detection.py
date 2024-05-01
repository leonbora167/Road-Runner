#uvicorn yolo_detection:yolo_app --host 0.0.0.0 --port 8000
import json
import requests
from fastapi import FastAPI
import base64
import uvicorn
import cv2 
import numpy as np
import torch 
import onnxruntime as ort 
import sys 
sys.path.append("../")
from helper_files.onnx_helpers import img_reshaping
from helper_files.onnx_helpers import onnx2xywh 



if torch.cuda.is_available():
    cuda=True
    print("GPU Available") 
else:
    cuda=False 
    print("GPU Not Available")  

model = "C:\git_projects\Road-Runner\weights\Yolov7_v2.onnx"
providers = ['CUDAExecutionProvider'] if cuda else ['CPUExecutionProvider']
session = ort.InferenceSession(model, providers=providers)
print(" YOLO ONNX Model Loaded ")

yolo_app = FastAPI()
sending_url = "http://127.0.0.1:8001/craft_detection"

@yolo_app.post("/")
def root():
    return {"message":"Welcome to YOLO Server"}

@yolo_app.post("/yolo_detection")
def yolo_predict(data: dict):
    #print("Inside The server")
    #Data is a dictionary containing "image", "metadata", "frame_number"
    frame_bytes = base64.b64decode(data["image"])
    img_frame = img_preprocessing(frame_bytes)
    orig_img = img_frame.copy()
    img, ratio, dwdh = img_reshaping(img_frame) #Preprocesses and returns an image array of size (1,3,640,640)
    outname = [i.name for i in session.get_outputs()]
    inname = [i.name for i in session.get_inputs()]
    inp = {inname[0]:img}
    outputs = session.run(outname, inp)[0]
    temp = onnx2xywh(outputs, orig_img, dwdh, ratio) #It returns a list which contains lists in the format of [[x1,y1,x2,y2],class id, confidence score]
    print(temp) #Print the coordinates and class values
    #print("FRAME NUMBER = ",data["frame_number"])
    counter = data["counter"]
    send_data(temp, orig_img, counter)
    print("send data function called")
    return {"Message ": "Data Received"}


def img_preprocessing(frame_bytes):
    frame = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
    return frame

def send_data(temp, orig_img, counter):
    for i in temp:
        x1,y1 = i[0][0], i[0][1]
        x2,y2 = i[0][2], i[0][3]
        class_id = i[1]
        conf_score = i[2]
        orig_img_bytes = cv2.imencode(".jpg", orig_img)[1].tobytes()
        orig_img_encoded = base64.b64encode(orig_img_bytes).decode("utf-8")  # Encode as Base64
        data = { "x1" : x1,
                 "x2" : x2,
                 "y1" : y1,
                 "y2" : y2,
                 "class": class_id,
                 "conf" : conf_score,
                 "full_frame" : orig_img_encoded}
        data["counter"] = counter
        packet = json.dumps(data)
        response = requests.post(url=sending_url, data=packet)
        print("Detections and Image Sent to CRAFT")

if __name__ == "__main__":
    uvicorn.run(yolo_app, host="0.0.0.0", port=8000)