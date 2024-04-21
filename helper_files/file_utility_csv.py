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

df = pd.DataFrame(columns=columns)
csv_file = "../output.csv"
df.to_csv(csv_file, index=False)
print("Output CSV File Created")

main_folder = "../test_db"
csv_file = "../output.csv"


def save_vehicle(data):
    df = pd.read_csv(csv_file)
    original_image_path = os.path.join("../test_data", str(data["counter"])+".jpg")
    frame_bytes = data["full_frame"]
    frame_array = bytes_to_frame(frame_bytes)
    cv2.imwrite(original_image_path, frame_array)
    cropped_image_path = os.path.join("../test_data", str(data["counter"])+"_cropped_1" + ".jpg")
    np_img_bytes = data["encoded_number_plate_image"]
    np_img_array = bytes_to_frame(np_img_bytes)
    cv2.imwrite(cropped_image_path, np_img_array)
    object_coordinates = str(data["x1"]) + "\t" + str(data["y1"]) + "\t" + str(data["x2"]) + "\t" + str(data["y2"])
    Class = str(1)
    ocr = " "
    Timestamp = str(datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'))
    data = [original_image_path, cropped_image_path, object_coordinates, Class, ocr, Timestamp]
    new_data = pd.DataFrame(data, columns=columns)
    new_df = pd.concat([df, new_data], ignore_index=True)
    new_df.to_csv(csv_file, index=False)

def save_number_plate(data):
    df = pd.read_csv(csv_file)
    original_image_path = os.path.join("../test_data", str(data["counter"])+".jpg")
    frame_bytes = data["full_frame"]
    frame_array = bytes_to_frame(frame_bytes)
    cv2.imwrite(original_image_path, frame_array)
    cropped_image_path = os.path.join("../test_data", str(data["counter"])+"_cropped_0" + ".jpg")
    np_img_bytes = data["encoded_number_plate_image"]
    np_img_array = bytes_to_frame(np_img_bytes)
    cv2.imwrite(cropped_image_path, np_img_array)
    object_coordinates = str(boxes)
    Class = str(0)
    ocr = " "
    Timestamp = str(datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'))
    data = [original_image_path, cropped_image_path, object_coordinates, Class, ocr, Timestamp]
    new_data = pd.DataFrame(data, columns=columns)
    new_df = pd.concat([df, new_data], ignore_index=True)
    new_df.to_csv(csv_file, index=False)




def bytes_to_frame(img_bytes): #Converts image from bytes to numpy array
    img_frame = base64.b64decode(img_bytes)
    img_frame = cv2.imdecode(np.frombuffer(img_frame, np.uint8), cv2.IMREAD_COLOR)
    return img_frame

def img_to_bytes(img_array): #Takes an array of an image and converts to bytest to wrap in a JSON Dict object
    orig_img_bytes = cv2.imencode(".jpg", img_array)[1].tobytes()
    orig_img_encoded = base64.b64encode(orig_img_bytes).decode("utf-8")  # Encode as Base64