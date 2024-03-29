import gdown
import os

#Download Yolov7 Trained and converted model weights to "weights" folder 
url = "https://drive.google.com/file/d/16ME8mKEtNQeQjRIK1tfA_y18-SAjJtcx/view?usp=sharing"
output = "weights/Yolov7_v2.onnx"
gdown.download(url=url, output=output, fuzzy=True)

url = "https://drive.google.com/file/d/1AEntTXE8uQvAFYW_LD5Ph3wWlsnCbLAE/view?usp=sharing"
output = "weights/Yolov7_v2.pt"
gdown.download(url=url, output=output, fuzzy=True)
