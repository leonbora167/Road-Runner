#Using git repo https://github.com/k9ele7en/ONNX-TensorRT-Inference-CRAFT-pytorch/tree/main

#Download weights from https://drive.google.com/open?id=1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ

#Put weights in folder C:\Projects\Road-Runner\conversion_files\ONNX-TensorRT-Inference-CRAFT-pytorch\weights

#pip install nvidia-pyindex
#pip install onnx-graphsurgeon

#Run the ONNX-TensorRT-Inference-CRAFT-pytorch\converters\pth2onnx.py
#Get the onnx weights from the weights folder

import os
import sys
import cv2
from helper_files.craft_onnx_helper import loadImage,normalizeMeanVariance, resize_aspect_ratio, getDetBoxes, adjustResultCoordinates
import torch
import numpy as np
import onnxruntime
import time
from imutils.perspective import four_point_transform
sys.path.append("../")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

#Default Argument Values
text_threshold=0.7
low_text=0.4
link_threshold=0.01
if torch.cuda.is_available():
    cuda=True
    device="cuda"
else:
    cuda=False
    device="cpu"
canvas_size=1100
mag_ratio=1.5
poly=False
show_time=False
test_folder=False
model_path = "..\weights\craft_onnx.onnx"

img = cv2.imread("../toll_test.jpg")
orig_img = img.copy()
img = loadImage(img)
img_resized, target_ratio, size_heatmap = resize_aspect_ratio(img, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio)
ratio_h = ratio_w = 1 / target_ratio
img_resized = normalizeMeanVariance(img_resized)
img_resized = torch.from_numpy(img_resized).permute(2, 0, 1)  # [h, w, c] to [c, h, w]

img_resized = (img_resized.unsqueeze(0))  # [c, h, w] to [b, c, h, w]
img_resized = img_resized.to(device)
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
ort_session = onnxruntime.InferenceSession(model_path)
print("CRAFT ONNX Model Loaded")

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img_resized)}
ort_outs = ort_session.run(None, ort_inputs)
y = ort_outs[0]

# make score and link map
score_text = y[0, :, :, 0]
score_link = y[0, :, :, 1]


# Post-processing
boxes, polys = getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

# coordinate adjustment
boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
polys = adjustResultCoordinates(polys, ratio_w, ratio_h)
for k in range(len(polys)):
    if polys[k ] is None: polys[k] = boxes[k]

#Draw Bounding Boxes for Craft Outputs
for i in boxes:
    x1,y1 = int(i[0][0]), int(i[0][1])
    x2,y2 = int(i[1][0]), int(i[1][1])
    x3,y3 = int(i[2][0]), int(i[2][1])
    x4,y4 = int(i[3][0]), int(i[3][1])
    cv2.rectangle(orig_img, (x1,y1), (x3,y3), (255,0,0), 2)
cv2.imwrite("../test_data/craft_test.jpg", orig_img)

#Save CRAFT Outputs to disk
counter = 0
for i in boxes:
    x1,y1 = int(i[0][0]), int(i[0][1])
    x2,y2 = int(i[1][0]), int(i[1][1])
    x3,y3 = int(i[2][0]), int(i[2][1])
    x4,y4 = int(i[3][0]), int(i[3][1])
    pts = np.array([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], dtype=np.float32)
    warped = four_point_transform(orig_img, pts)
    cv2.imwrite("../test_data/craft_img_"+str(counter)+".jpg", warped)
