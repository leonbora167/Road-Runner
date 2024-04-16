import gdown
import os

os.system("mkdir test_data")
os.system("mkdir weights")

#Download Yolov7 Trained and converted model weights to "weights" folder 
url = "https://drive.google.com/file/d/16ME8mKEtNQeQjRIK1tfA_y18-SAjJtcx/view?usp=sharing"
output = "weights/Yolov7_v2.onnx"
gdown.download(url=url, output=output, fuzzy=True)

url = "https://drive.google.com/file/d/1AEntTXE8uQvAFYW_LD5Ph3wWlsnCbLAE/view?usp=sharing"
output = "weights/Yolov7_v2.pt"
gdown.download(url=url, output=output, fuzzy=True)

#CCTV Footage from web for testing, Attributed link given below 
#https://www.videvo.net/video/cars-driving-along-an-indian-freeway/6374/
url = "https://drive.google.com/file/d/1o3brw1nj8_BnZlOPmPU0-RBDqDRVnZZw/view?usp=sharing"
output = "test_data/test_video_1.mp4"
gdown.download(url=url, output=output, fuzzy=True)

#Video found on Youtube 
#https://www.youtube.com/watch?v=G-ie5hQbG2s
url = "https://drive.google.com/file/d/1NpBp7zwPT1zF3r1oO57a_Kf1ZcnZxECE/view?usp=sharing"
output = "test_data/test_video_2.mp4"
gdown.download(url=url, output=output, fuzzy=True)

#Download CRAFT -> ONNX weights
url = "https://drive.google.com/file/d/1dfy3TMEmEkuJUBC7Nw0kp7grUZ7hcL3k/view?usp=sharing"
output = "weights/craft_onnx.onnx"
gdown.download(url=url, output=output, fuzzy=True)