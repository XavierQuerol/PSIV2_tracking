from tracker import Tracker
#from ultralytics import YOLO
import cv2
import numpy as np
import imageio
import os
from detect import *
from tracker import *

#OPTIONS: c_final, one_line, two_lines
tracker = Tracker("c_final")

detect_op = 'morpho' #opcions: 'morpho', 'yolo'

if detect_op == 'morpho':
    detect = Detect()
elif detect_op == 'yolo':
    model = YOLO('yolov8n.pt')


# Perform tracking with the model
video_path = "output2.mp4"
cap = cv2.VideoCapture(video_path)
i = 0
frames = []

fgbg = cv2.createBackgroundSubtractorMOG2()

# Read a frame from the video
success, frame = cap.read()

# Loop through the video frames
while cap.isOpened():
    if success and i%3 == 0:
        # Read a frame from the video
        success, frame = cap.read()

        if detect_op == 'morpho':
            #crida delimitar zona 
            img = detect.delimitar_zona(frame)
            #crida prepro 
            img = detect.prepro(img, fgbg)
            #crida trobar cotxes 
            bboxes = detect.trobar_contorns(img)
            cv2.imshow('morpho',img)

        elif detect_op == 'yolo':
            area_pts = np.array([[200,360], [380,360], [450,960], [50,960]])
            imAux = np.zeros(shape=(frame.shape[:2]), dtype=np.uint8)
            imAux = cv2.drawContours(imAux, [area_pts], -1, (255), -1)
            image_area = cv2.bitwise_and(frame, frame, mask=imAux)
            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            results = model.predict(image_area, conf=0.4)[0]
            bboxes = []
            if results.bboxes.shape[0] !=0:
                for box in range(results.bboxes.shape[0]):
                    if results.bboxes.cls[box] == 2:
                        bboxes.append(results.bboxes.xyxy[box,:].tolist())

        bboxes = np.array(bboxes)
        tracker.recalculate(bboxes, frame)


    i+=1