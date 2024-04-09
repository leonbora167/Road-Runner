import zmq 
import base64
import json 
import cv2
import numpy as np
import requests

SERVER_ADDRESS = "127.0.0.1"
SERVER_PORT = 8000
SERVICE_NAME = "yolo_detection"
url = f"http://{SERVER_ADDRESS}:{SERVER_PORT}/{SERVICE_NAME}"

def run():
    context = zmq.Context() 
    socket = context.socket(zmq.SUB)
    socket.subscribe("") #Subscribe to all topics 
    socket.connect("tcp://localhost:5555")
    while True:
        data_json = socket.recv_json()
        data = data_json
        headers = {"Content-Type": "application/json"}
        print("Type of data being sent is ",type(data))
        #url = f"http://{SERVER_ADDRESS}:{SERVER_PORT}/{SERVICE_NAME}/{json.dumps(data)}"
        response = requests.post(url=url, data=json.dumps(data))
        #response.raise_for_status()
        print("Image Sent to YOLO")

def main():
    run()

if __name__ == "__main__":
    main()