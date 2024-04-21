#uvicorn craft_detection:craft_app --host 0.0.0.0 --port 8001
import sys
sys.path.append("../")
from fastapi import FastAPI
import numpy as np
import cv2
import torch
import base64
from helper_files.craft_onnx_helper import loadImage,normalizeMeanVariance, resize_aspect_ratio, getDetBoxes, adjustResultCoordinates
from imutils.perspective import four_point_transform
import json
import onnxruntime
from helper_files.file_utility_csv import save_vehicle, save_number_plate



craft_app = FastAPI()
@craft_app.post("/")
def root():
    return {"message":"Welcome to YOLO Server"}

@craft_app.post("/craft_detection")
def craft_predict(data: dict):
    print("X1 Coordinates are ",data["x1"])
    craft_infer(data)
    return {"Message :" "YOLO DATA Received"}

def bytes_to_frame(img_bytes):
    img_frame = base64.b64decode(img_bytes)
    img_frame = cv2.imdecode(np.frombuffer(img_frame, np.uint8), cv2.IMREAD_COLOR)
    return img_frame

def craft_infer(data):
    full_frame = bytes_to_frame(data["full_frame"]) #Image in array format
    class_id = data["class"] #Class id              0 | Number Plate      ;         1 | Vehicle
    yolo_crop = full_frame[y1:y2, x1:x2]  # Cropped image taken from YOLO Model
    if(class_id == 1):
        encoded_vehicle_image = img_to_bytes(yolo_crop)
        data["encoded_vehicle_image"] = encoded_vehicle_image
        save_vehicle(data)
        #break
    elif(class_id == 0)
        #x1,y1, x2,y2 = data["x1"], data["y1"], data["x2"], data["y2"]
        img_resized, ratio_w, ratio_h = craft_preprocessing(yolo_crop)
        ort_inputs = {model.get_inputs()[0].name: to_numpy(img_resized)}
        ort_outs = model.run(None, ort_inputs)
        y = ort_outs[0]

        # make score and link map
        score_text = y[0, :, :, 0]
        score_link = y[0, :, :, 1]

        # Post-processing
        boxes, polys = getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

        # coordinate adjustment
        boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
        data["text_segment_coordinates"] = boxes
        encoded_number_plate_image = img_to_bytes(yolo_crop)
        data["encoded_number_plate_image"] = encoded_number_plate_image
        save_number_plate(data)
        #cv2.imwrite("../temp/"+str(x1)+".jpg", yolo_crop) #Was checking if the outputs are coming or not
        #print("Boxes are \n",boxes)

def load_craft():
    model_path = "..\weights\craft_onnx.onnx"
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
    ort_session = onnxruntime.InferenceSession(model_path)
    print("CRAFT ONNX Model Loaded")
    return ort_session

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def craft_preprocessing(img):
    img = loadImage(img)
    img_resized, target_ratio, size_heatmap = resize_aspect_ratio(img, canvas_size, interpolation=cv2.INTER_LINEAR,
                                                                  mag_ratio=mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio
    img_resized = normalizeMeanVariance(img_resized)
    img_resized = torch.from_numpy(img_resized).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
    img_resized = (img_resized.unsqueeze(0))  # [c, h, w] to [b, c, h, w]
    img_resized = img_resized.to(device)
    return img_resized, ratio_w, ratio_h


# Global Parameters
text_threshold = 0.7
low_text = 0.4
link_threshold = 0.01
if torch.cuda.is_available():
    cuda = True
    device = "cuda"
else:
    cuda = False
    device = "cpu"
canvas_size = 1100
mag_ratio = 1.5
poly = False
show_time = False
test_folder = False

#Loading CRAFT Model in ONNX Format
model = load_craft()

def img_to_bytes(img_array): #Takes an array of an image and does
    img_bytes = cv2.imencode(".jpg", img_array)[1].tobytes()
    img_encoded = base64.b64encode(img_bytes).decode("utf-8")  # Encode as Base64
    return img_encoded