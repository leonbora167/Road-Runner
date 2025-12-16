import zmq
import json 
import threading 
import cv2 
import numpy as np 
from flask import Flask, jsonify 
from ultralytics import YOLO 
from helper_files.yolo_ultralytics import yolo_postprocess_np

app = Flask(__name__)

model = YOLO(model = r"C:\Projects\Road-Runner\weights\yolov11s_np.pt")
print("YOLO Model for Vehicle Detection loaded")


def zmq_listener():
    context = zmq.Context() 

    sub_socket_1 = context.socket(zmq.SUB)
    sub_socket_1.connect("tcp://localhost:5556")
    sub_socket_1.setsockopt_string(zmq.SUBSCRIBE, "frame") #Subscribing to "frame" topic only
    print("Yolo Number Plate [SUB] Listening for frames")

    sub_socket_2 = context.socket(zmq.SUB)
    sub_socket_2.connect("tcp://localhost:5556")
    sub_socket_2.setsockopt_string(zmq.SUBSCRIBE, "yolo_vehicle") # Subscribing to "yolo_vehicle" topic only 
    print("Yolo Number Plate [SUB] listening for yolo-vehicle detections")

    pub_socket = context.socket(zmq.PUB)
    pub_socket.bind("tcp://*:5557")
    print("Yolo Number Plate [PUB] Ready to infer and send responses")

    while True:
        topic, metadata_json, frame_bytes = sub_socket_1.recv_multipart() #For original frame + metadata
        metadata = json.loads(metadata_json.decode("utf-8"))

        #Decode image
        frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

        topic, yolo_vehicle_json = sub_socket_2.recv_multipart() # For vehicle detection outputs 
        yolo_vehicle_json = json.loads(yolo_vehicle_json.decode("utf-8"))
        detections = yolo_vehicle_json["yolo_response"] # [[x1,y1,x2,y2,class_id,confidence]]
        for det in detections:
            x1,y1,x2,y2, class_id, confidence = det
            vehicle_image = frame[y1:y2, x1:x2]
            results = model.predict(vehicle_image, device = 0, verbose=False)
            model_response = yolo_postprocess_np(results) # [[x1,y1,x2,y2,class_id,confidence]]
            print(f"Inferencing done for NP")
            # model_response = model_response[0] #Taking the assumption for one number plate per vehicle so single list only -> [x1,y1,x2,y2, class_id, confidence]

            message_1 = {
                "frame_id" : metadata["frame_id"],
                "timestamp" : metadata["timestamp"],
                "frame_shape" : metadata["frame_shape"]
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
                b"yolo_number_plate", #Topic
                json.dumps(message_2).encode("utf-8") 
            ])


@app.route("/status", methods = ["GET"])
def status():
    return jsonify({"status":"YOLO Number Plate server up"})

if __name__ == "__main__":
    #Start ZMQ listener in bg
    thread = threading.Thread(target=zmq_listener, daemon=True) #daemon=True ensures when flask exits the zmq thread exits too automatically preventing zombie processes
    thread.start() 
    app.run(host = "0.0.0.0", port=8001, debug=False)