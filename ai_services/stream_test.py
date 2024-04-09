import zmq 
import base64
import json 
import cv2
import numpy as np

context = zmq.Context() 
socket = context.socket(zmq.SUB)
socket.subscribe("") #Subscribe to all topics 
socket.connect("tcp://localhost:5555")

while True:
    data_json = socket.recv_json()
    #data = json.loads(data_json)
    data = data_json
    
    frame_bytes = base64.b64decode(data["image"])
    frame = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
    metadata = data["metadata"]

    print("Frame shape is ", frame.shape)
    print("Counter is ", data["frame_number"])