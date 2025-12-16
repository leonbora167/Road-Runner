import cv2 
import zmq 
import json 
import time 

video_path = "data/test_video_1.mp4"
zmq_port = 5555

context = zmq.Context() 
socket = context.socket(zmq.PUB)
socket.bind(f"tcp://localhost:{zmq_port}") # even f"tcp://*:{zmq_port}" works

print(f"[PUB] Publishing frames on port {zmq_port}")

cap = cv2.VideoCapture(video_path)
frame_idx = 0 

while cap.isOpened():
    ret, frame = cap.read() 
    if not ret:
        break 

    _, buffer = cv2.imencode(".jpg", frame) #Encode frame as JPEG
    frame_bytes = buffer.tobytes() 

    metadata = {
        "frame_id" : frame_idx,
        "timestamp" : time.time(),
        "img_shape" : frame.shape 
    }

    #send message 
    socket.send_multipart([
        b"frame", #Topic of message
        json.dumps(metadata).encode("utf-8"),
        frame_bytes
    ])

    frame_idx += 1 
    #time.sleep(0.3) #Waiting to stimulate 30 fps but will test later
    print(f"Frame [{frame_idx} sent to Vehicle Detector]")

cap.release()