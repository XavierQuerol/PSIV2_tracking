from tracker import Tracker
from ultralytics import YOLO
import cv2
import numpy as np
import imageio
import os

#OPTIONS: c_final, one_line, two_lines
tracker = Tracker("cfinal")


# Load an official or custom model
model = YOLO('yolov8n.pt')

# Perform tracking with the model

video_path = "output2.mp4"
cap = cv2.VideoCapture(video_path)
i = 0
frames = []

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    area_pts = np.array([[200,360], [380,360], [450,960], [50,960]])
    imAux = np.zeros(shape=(frame.shape[:2]), dtype=np.uint8)
    imAux = cv2.drawContours(imAux, [area_pts], -1, (255), -1)
    image_area = cv2.bitwise_and(frame, frame, mask=imAux)
    if success and i%3 == 0:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.predict(image_area, conf=0.4)[0]
        boxes = []
        if results.boxes.shape[0] !=0:
            for box in range(results.boxes.shape[0]):
                if results.boxes.cls[box] == 2:
                    boxes.append(results.boxes.xyxy[box,:].tolist())

        boxes = np.array(boxes)
        tracker.recalculate(boxes, frame)

    if i == 703:
        break

    i+=1