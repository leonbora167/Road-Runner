import zmq
import json 
import threading 
import cv2 
import numpy as np 
from flask import Flask, jsonify 
from ultralytics import YOLO 
from helper_files.yolo_ultralytics import yolo_postprocess_np
import logging, os

app = Flask(__name__)

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "yolo_np.log")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    handlers=[
                        logging.FileHandler(log_file, mode="a")
                        ])

logging.info("YOLO Number Plate Service started")

model = YOLO(model = r"weights//yolov11s_np.pt")
print("YOLO Model for Vehicle Detection loaded")


def zmq_listener():
    context = zmq.Context() 

    sub_socket_1 = context.socket(zmq.SUB)
    sub_socket_1.connect("tcp://localhost:5557")
    sub_socket_1.setsockopt_string(zmq.SUBSCRIBE, "frame") #Subscribing to "frame" topic only
    print("Yolo Number Plate [SUB] Listening for frames")

    sub_socket_2 = context.socket(zmq.SUB)
    sub_socket_2.connect("tcp://localhost:5557")
    sub_socket_2.setsockopt_string(zmq.SUBSCRIBE, "vehicle_tracker") # Subscribing to "yolo_vehicle" topic only 
    print("Yolo Number Plate [SUB] listening for yolo-vehicle detections")

    pub_socket = context.socket(zmq.PUB)
    pub_socket.bind("tcp://*:5558")
    print("Yolo Number Plate [PUB] Ready to infer and send responses")

    while True:
        topic, metadata_json, frame_bytes = sub_socket_1.recv_multipart() #For original frame + metadata
        metadata = json.loads(metadata_json.decode("utf-8"))

        #Decode image
        frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

        topic, vehicle_tracker_json = sub_socket_2.recv_multipart() # For vehicle detection outputs 
        vehicle_tracker_json = json.loads(vehicle_tracker_json.decode("utf-8"))
        detections = vehicle_tracker_json["tracker_outputs"] # [[x1,y1,x2,y2, conf, class_id, track_id], ....] 
        for det in detections:
            x1,y1,x2,y2, tracker_conf, class_id, track_id  = det
            vehicle_image = frame[y1:y2, x1:x2]
            results = model.predict(vehicle_image, device = 0, verbose=False)
            try:
                model_response, np_image = yolo_postprocess_np(results, vehicle_image) # [[x1,y1,x2,y2,class_id,confidence]]
            except:
                continue
            #print(f"Inferencing done for NP")
            logging.info(f"NP Inferencing done for Vehicle : {track_id}")

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

            _, np_image_buffer = cv2.imencode(".jpg", np_image) #Encode frame as JPG
            np_image_bytes = np_image_buffer.tobytes()
            message_2 = {
                "yolo_outputs" : model_response,
                "track_id" : track_id 
            }
            pub_socket.send_multipart([
                b"yolo_number_plate", #Topic
                json.dumps(message_2).encode("utf-8"),
                np_image_bytes 
            ])


@app.route("/status", methods = ["GET"])
def status():
    return jsonify({"status":"YOLO Number Plate server up"})

if __name__ == "__main__":
    #Start ZMQ listener in bg
    thread = threading.Thread(target=zmq_listener, daemon=True) #daemon=True ensures when flask exits the zmq thread exits too automatically preventing zombie processes
    thread.start() 
    app.run(host = "0.0.0.0", port=8002, debug=False)