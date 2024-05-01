import csv
import os.path
import sys
sys.path.append("../")
import cv2
import numpy as np
import base64
import pandas as pd
import time
import datetime


ts = time.time()
columns = ["Original Image Path", "Cropped Image Path",
           "Object Coordinates", "Class", "OCR", "Timestamp"]

output_file = "../output.txt"


def save_vehicle(data):
    original_image_path = os.path.join("../output_data", str(data["counter"])+".jpg")
    frame_bytes = data["full_frame"]
    frame_array = bytes_to_frame(frame_bytes)
    #cv2.imwrite(original_image_path, frame_array)
    cropped_image_path = os.path.join("../output_data", str(data["counter"])+"_cropped_1" + ".jpg")
    try:
        np_img_bytes = data["encoded_number_plate_image"]
        np_img_array = bytes_to_frame(np_img_bytes)
        cv2.imwrite(cropped_image_path, np_img_array)
    except:
        pass
    object_coordinates = str(data["x1"]) + "\t" + str(data["y1"]) + "\t" + str(data["x2"]) + "\t" + str(data["y2"])
    Class = str(1)
    ocr = " "
    Timestamp = str(datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'))
    data = str(original_image_path) + "\t" +str(cropped_image_path) +"\t" + str(object_coordinates) + "\t" + str(Class) + "\t" + str(ocr) + "\t" + str(Timestamp)
    with open(output_file, "a") as x:
        x.write(data)

def save_number_plate(data):
    original_image_path = os.path.join("../output_data", str(data["counter"])+".jpg")
    frame_bytes = data["full_frame"]
    frame_array = bytes_to_frame(frame_bytes)
    #cv2.imwrite(original_image_path, frame_array) #Saves the original frames without any cropping on detections
    cropped_image_path = os.path.join("../output_data", str(data["counter"])+"_cropped_0" + ".jpg")
    try:
        np_img_bytes = data["encoded_number_plate_image"]
        np_img_array = bytes_to_frame(np_img_bytes)
        cv2.imwrite(cropped_image_path, np_img_array)
    except:
        pass
    boxes = data["text_segment_coordinates"]
    object_coordinates = str(boxes)
    Class = str(0)
    ocr = " "
    Timestamp = str(datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'))
    data = str(original_image_path) + "\t" +str(cropped_image_path) +"\t" + str(object_coordinates) + "\t" + str(Class) + "\t" + str(ocr) + "\t" + str(Timestamp)
    with open(output_file, "a") as x:
        x.write(data)





def bytes_to_frame(img_bytes): #Converts image from bytes to numpy array
    img_frame = base64.b64decode(img_bytes)
    img_frame = cv2.imdecode(np.frombuffer(img_frame, np.uint8), cv2.IMREAD_COLOR)
    return img_frame

def img_to_bytes(img_array): #Takes an array of an image and converts to bytest to wrap in a JSON Dict object
    orig_img_bytes = cv2.imencode(".jpg", img_array)[1].tobytes()
    orig_img_encoded = base64.b64encode(orig_img_bytes).decode("utf-8")  # Encode as Base64
    return orig_img_encoded