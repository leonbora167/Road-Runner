from paddleocr import TextDetection, TextRecognition
import cv2
import logging, os 
from imutils.perspective import four_point_transform 
import numpy as np
import zmq
import json 
import threading 
from flask import Flask, jsonify 
import requests

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "yolo_np.log")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    handlers=[
                        logging.FileHandler(log_file, mode="a")
                        ])

os.environ["PADDLE_PDX_MODEL_SOURCE"] = "BOS"

det_model = TextDetection(model_name = "PP-OCRv5_mobile_det")
rec_model = TextRecognition(model_name = "PP-OCRv5_mobile_rec")
logging.info("Detector and Recognizer from PaddleOCR Loaded")
app = Flask(__name__)

def zmq_listener():
    context = zmq.Context() 

    sub_socket_1 = context.socket(zmq.SUB)
    sub_socket_1.connect("tcp://localhost:5558")
    sub_socket_1.setsockopt_string(zmq.SUBSCRIBE, "frame") #Subscribing to "frame" topic only
    print("PaddleOCR [SUB] Listening for frames")

    sub_socket_2 = context.socket(zmq.SUB)
    sub_socket_2.connect("tcp://localhost:5558")
    sub_socket_2.setsockopt_string(zmq.SUBSCRIBE, "yolo_number_plate") # Subscribing to "yolo_vehicle" topic only 
    print("PaddleOCR [SUB] listening for yolo-vehicle detections")

    pub_socket = context.socket(zmq.PUB)
    pub_socket.bind("tcp://*:5559")
    print("PaddleOCR [PUB] Ready to infer and send responses")

    while True:
        topic, metadata_json, frame_bytes = sub_socket_1.recv_multipart() #For original frame + metadata
        frame_metadata = json.loads(metadata_json.decode("utf-8"))
        frame_id = frame_metadata["frame_id"]
        timestamp = frame_metadata["timestamp"]
        #Decode image
        frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame_img = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

        topic, number_plate_json, number_plate_bytes = sub_socket_2.recv_multipart() # For vehicle detection outputs 
        number_plate_json = json.loads(number_plate_json.decode("utf-8"))
        vehicle_track_id = number_plate_json["track_id"]
        number_plate_array = np.frombuffer(number_plate_bytes, dtype=np.uint8)
        number_plate_img = cv2.imdecode(number_plate_array, cv2.IMREAD_COLOR)

        det_outputs = det_model.predict(number_plate_img, batch_size=1)
        for index, i in enumerate(det_outputs[0]["dt_polys"]):
            x1, y1 = i[0][0], i[0][1]
            x2, y2 = i[1][0], i[1][1]
            x3, y3 = i[2][0], i[2][1]
            x4, y4 = i[3][0], i[3][1]
            pts = np.array([[x1,y1], [x2,y2], [x3,y3], [x4,y4]])
            warped = four_point_transform(number_plate_img, pts)
            rec_output = rec_model.predict(input=warped, batch_size=1)
            rec_output = rec_output[0]
            rec_confidence = rec_output["rec_score"]
            rec_text = rec_output["rec_text"]
            #print("Text is ", rec_text)  
            payload = {"frame_id" : frame_id,
                       "timestamp" : timestamp,
                       "vehicle_track_id" : vehicle_track_id,
                       "rec_confidence" : rec_confidence,
                       "rec_text" : rec_text}
            try:
                requests.post("http://localhost:8004/ingest", json=payload, timeout=0.5)
            except Exception as e:
                logging.error(f"OCR ingest failed: {e}")       


@app.route("/status", methods = ["GET"])
def status():
    return jsonify({"status":"OCR server up"})

if __name__ == "__main__":
    #Start ZMQ listener in bg
    thread = threading.Thread(target=zmq_listener, daemon=True) #daemon=True ensures when flask exits the zmq thread exits too automatically preventing zombie processes
    thread.start() 
    app.run(host = "0.0.0.0", port=8003, debug=False)



