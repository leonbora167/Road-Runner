import zmq
import json 
import threading 
import cv2 
import numpy as np 
from flask import Flask, jsonify 
from ultralytics import YOLO 
from helper_files.yolo_ultralytics import yolo_postprocess_vehicle

app = Flask(__name__)

model = YOLO(model = r"C:\Projects\Road-Runner\weights\yolo11s.pt")
print("YOLO Model for Vehicle Detection loaded")


def zmq_listener():
    context = zmq.Context() 

    sub_socket = context.socket(zmq.SUB)
    sub_socket.connect("tcp://localhost:5555")
    sub_socket.setsockopt_string(zmq.SUBSCRIBE, "frame") #Subscribing to "frame" topic only
    print("Yolo Vehicle [SUB] Listening for frames")

    pub_socket = context.socket(zmq.PUB)
    pub_socket.bind("tcp://*:5556")
    print("Yolo Vehicle [PUB] Ready to infer and send responses")

    while True:
        topic, metadata_json, frame_bytes = sub_socket.recv_multipart()
        metadata = json.loads(metadata_json.decode("utf-8"))

        #Decode image
        frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                                                                                                                                                  
        results = model.predict(frame, device = 0, verbose=False)
        model_response = yolo_postprocess_vehicle(results) #[[x1,y1,x2,y2,class_id,confidence]]   #List of list since I am thinking of sending multiple vehicles to tracker                                                        
        if model_response == []: #Assuming no vehicles in frame or no valid vehicle dets in frame                                      
            continue    #Keep on trying frames till you get one with correct vehicle dets                                                                                                 
        print(f"Inferencing done for Frame ID [{metadata['frame_id']}]")                                                     
                                                                                                                       
        message_1 = {                                                                                      
            "frame_id" : metadata["frame_id"],
            "timestamp" : metadata["timestamp"],
            "frame_shape" : metadata["img_shape"]
        }
        pub_socket.send_multipart([
            b"frame", #Topic of message
            json.dumps(message_1).encode("utf-8"),
            frame_bytes
            ])                                                                      

        message_2 = {
            "yolo_response" : model_response 
        }
        pub_socket.send_multipart([
            b"yolo_vehicle", #Topic
            json.dumps(message_2).encode("utf-8") 
        ]) 

@app.route("/status", methods = ["GET"])
def status():
    return jsonify({"status":"YOLO Vehicle server up"})

if __name__ == "__main__":
    #Start ZMQ listener in bg
    thread = threading.Thread(target=zmq_listener, daemon=True) #daemon=True ensures when flask exits the zmq thread exits too automatically preventing zombie processes
    thread.start()
    app.run(host = "0.0.0.0", port=8000, debug=False)