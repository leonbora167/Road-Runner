#uvicorn yolo_detection:yolo_app --host 0.0.0.0 --port 8000
from fastapi import FastAPI 
import base64
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
    temp = onnx2xywh(outputs, orig_img, dwdh, ratio)
    print(temp) #Print the coordinates and class values
    print("FRAME NUMBER = ",data["frame_number"])
    #data = await request.json()
    return {"Message ": "Data Received"}


def img_preprocessing(frame_bytes):
    frame = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
    return frame 