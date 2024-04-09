import cv2 
import numpy as np 
import json 
import time 
import argparse
import zmq
import base64


#Video Capture
def load_stream(args):
    if(args.input.split(".")[-1]=="mp4"): #MP4 Video Stream format
        print("Path of video file is ", args.input)
        cap = cv2.VideoCapture(args.input)
        return cap

#ZMQ Video Streaming
def video_streaming(cap, args):
    frame_counter = 0
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind("tcp://127.0.0.1:5555")
    print("Socket Connection Established")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_counter += 1
        frame_bytes = cv2.imencode(".jpg", frame)[1].tobytes()
        encoded_image = base64.b64encode(frame_bytes).decode("utf-8") #Encode as Base64
        metadata = "placeholder"

        data = {
            "image": encoded_image,
            "metadata": metadata,
            "frame_number": frame_counter
        }

        #Send data as JSON over ZMQ 
        socket.send_json(data)
    cap.release()



def main():
    #print("Inside main function")
    parser = argparse.ArgumentParser(description="Argument for the video streaming")
    parser.add_argument("--input", type=str, required=True, help="path to video file or rtsp stream")
    parser.add_argument("--output", type=str, help="path to output file for csv")
    args = parser.parse_args()

    cap = load_stream(args)

    video_streaming(cap, args)


if __name__ == "__main__":
    main()