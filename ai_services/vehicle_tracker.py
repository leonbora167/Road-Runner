import zmq
import json 
import threading 
import cv2 
import numpy as np 
from flask import Flask, jsonify 
from helper_files.vision_models import yolo_to_boxmot, tracks_xyxy
from boxmot import OcSort
#from boxmot import BotSort, ByteTrack, OcSort
# import torch
# print(torch.cuda.is_available())

app = Flask(__name__)

tracker = OcSort(reid_weights = "osnet_x0_25",
                        device="cuda:0",
                        half=False)
print("Tracker Loaded")

def zmq_listener():
    context = zmq.Context() 

    sub_socket_1 = context.socket(zmq.SUB)
    sub_socket_1.connect("tcp://localhost:5556")
    sub_socket_1.setsockopt_string(zmq.SUBSCRIBE, "frame") #Subscribing to "frame" topic only
    print("Tracker [SUB] Listening")

    sub_socket_2 = context.socket(zmq.SUB)
    sub_socket_2.connect("tcp://localhost:5556")
    sub_socket_2.setsockopt_string(zmq.SUBSCRIBE, "yolo_vehicle") #Subscribing to "yolo vehicle outputs" topic only
    print("Tracker [SUB] Listening")

    pub_socket = context.socket(zmq.PUB)
    pub_socket.bind("tcp://*:5557")
    print("Vehicle Tracker [PUB] ready for infer to Number Plate")

    while True:
        topic, metadata_json, frame_bytes = sub_socket_1.recv_multipart()
        frame_metadata = json.loads(metadata_json.decode("utf-8"))
        
        #Decode image
        frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

        topic, yolo_metadata = sub_socket_2.recv_multipart() 
        yolo_metadata = json.loads(yolo_metadata.decode("utf-8"))

        vehicle_detections_list = yolo_metadata["yolo_response"]
        vehicle_detections = yolo_to_boxmot(vehicle_detections_list)
        tracks = tracker.update(vehicle_detections,frame) #List of lists. Each list will contain [x1, y1, x2, y2, track_id, score, class_id, det_index]
        absolute_tracks = tracks_xyxy(tracks) # [[x1,y1,x2,y2, conf, class_id, track_id], ....] 

        message_1 = {                                                                                      
            "frame_id" : frame_metadata["frame_id"],
            "timestamp" : frame_metadata["timestamp"],
            "frame_shape" : frame_metadata["frame_shape"]
        }
        pub_socket.send_multipart([
            b"frame", #Topic of message
            json.dumps(message_1).encode("utf-8"),
            frame_bytes
            ])                                                                      

        message_2 = {
            "tracker_outputs" : absolute_tracks 
        }
        pub_socket.send_multipart([
            b"vehicle_tracker", #Topic
            json.dumps(message_2).encode("utf-8") 
        ]) 
        print(f"Inferencing for frame {frame_metadata['frame_id']}")

@app.route("/status", methods = ["GET"])
def status():
    return jsonify({"status":"Vehicle Tracker server up"})

if __name__ == "__main__":
    #Start ZMQ listener in bg
    thread = threading.Thread(target=zmq_listener, daemon=True) #daemon=True ensures when flask exits the zmq thread exits too automatically preventing zombie processes
    thread.start()
    app.run(host = "0.0.0.0", port=8001, debug=False)
                                                                                                                                                  