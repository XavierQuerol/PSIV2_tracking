from tracker import Tracker
from ultralytics import YOLO
import cv2
import numpy as np

#OPTIONS: c_final, one_line, two_lines
tracker = Tracker("c_final")


# Load an official or custom model
model = YOLO('yolov8n.pt')

# Perform tracking with the model

video_path = "output2.mp4"
cap = cv2.VideoCapture(video_path)
i = 0


# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    if success and i%3 == 0:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.predict(frame, conf=0.4)[0]
        boxes = []
        if results.boxes.shape[0] !=0:
            for box in range(results.boxes.shape[0]):
                if results.boxes.cls[box] == 2:
                    boxes.append(results.boxes.xyxy[box,:].tolist())

        boxes = np.array(boxes)
        tracker.recalculate(boxes, frame)

    i+=1