# Road-Runner

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

## Open Source ANPR Platform

- Goal is to create an open source anpr platform that will be able to work on a video stream and manage both video stream balancing and reconciliation. 
- A robust anpr platform consist of each module performing upto the standard as per the video type and lightning conditions. While the number detector could be invariant to training data of any country, for vehicle detection it would be recommended to train on the images of vehicles you would be implementing your vide on.

---
###  Breakdown

```md
| Service | Use | 
|-------|--------|
| app.py | Runs the UI along with setting up the model services and db config servers |
| stream_handler.py | Publishes Frames to TCP Port |
| yolo_vehicle.py | Vehicle detection using pretrained yolo_v11s |
| vehicle_tracker.py | Vehicle tracking using Boxmot |
| yolo_number_plate.py | Number plate detection using finetuned yolo_v11S |
| ocr1.py | Does text detection and recognition using paddleocr-v3 with two separate modules |
| data_utilities.py | Service to ingest data and create normalized values |
| reconcile.py | Helper script to take the normalized values and reconcile to create the final Vehicle_ID : Number Plate Value |
```

---
## Streamlit UI
![Streamlit](https://img.shields.io/badge/Streamlit-%23FE4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white)

![Application UI](data/version1_ui.gif)

```python
streamlit run app.py
```

