from ultralytics import YOLO
import cv2
import numpy as np
from shapely.geometry import LineString, Polygon

class Cotxe():
    def __init__(self, x, y, w, h, id):
        self.centroide = [x,y]
        self.shape = [w,h]
        self.id = id
        self.frames = 0
        self.show = True
        self.up = False
        self.down = False
        self.TROBAT = False
        self.trajec = [(x,y)]



class Tracker():

    def __init__(self):
        self.cotxes = []
        self.nou_id = 0
        self.COTXES_PUGEN = 0
        self.COTXES_BAIXEN = 0

    def calculate_distances(self, bboxes, frame):
        
        if len(self.cotxes) == 0:
            for i in range(bboxes.shape[0]):
                self.nou_id += 1
                id = self.nou_id
                cotxe = Cotxe(x=bboxes[i,0], y=bboxes[i,1], w=bboxes[i,2], h=bboxes[i,3], id=id)
                self.cotxes.append(cotxe)
        else:

            for cotxe in self.cotxes:
                cotxe.show = False

            cotxes_actuals = np.zeros((len(self.cotxes), 2))
            cotxes_nous = bboxes[:,:2]
            for i in range(len(self.cotxes)):
                cotxes_actuals[i,:] = self.cotxes[i].centroide


            distances = np.zeros((cotxes_nous.shape[0], cotxes_actuals.shape[0]))

            for i in range(cotxes_nous.shape[0]):
                for j in range(cotxes_actuals.shape[0]):
                    distances[i,j] = np.linalg.norm(cotxes_nous[i,:]-cotxes_actuals[j,:])

            maxims = distances.argmin(axis=0)
            maxims2 = distances.argmin(axis=1)

            for i in range(len(maxims2)):
                j = maxims2[i]
                if maxims[j] == i and distances[i, j] < 80:
                    self.cotxes[j].centroide = bboxes[i,:2]
                    self.cotxes[j].shape = bboxes[i,2:]
                    self.cotxes[j].show = True
                    self.cotxes[j].trajec.append((bboxes[i,:2][0], bboxes[i,:2][1]))
                else:
                    self.nou_id += 1
                    id = self.nou_id
                    cotxe = Cotxe(x=bboxes[i,0], y=bboxes[i,1], w=bboxes[i,2], h=bboxes[i,3], id=id)
                    self.cotxes.append(cotxe)
        cotxes_delete = []       
        for i, cotxe in enumerate(self.cotxes):
            if cotxe.show==True:
                cotxe.frames = 0
                frame = cv2.putText(frame, str(cotxe.id), (int(cotxe.centroide[0]), int(cotxe.centroide[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 255, 2)
                frame = cv2.rectangle(frame, (int(cotxe.centroide[0]),int(cotxe.centroide[1])), (int(cotxe.shape[0]), int(cotxe.shape[1])), 255, 5)
                    
            else:
                cotxe.frames += 1
                if cotxe.frames >5:
                    if (cotxe.trajec[0] - cotxe.trajec[-1]) > 50:
                        self.COTXES_PUGEN += 1
                    elif (cotxe.trajec[0] - cotxe.trajec[-1]) < -50:
                        self.COTXES_BAIXEN += 1
                    cotxes_delete.append(i)
                    
            self.cotxes = [cotxe for i, cotxe in enumerate(self.cotxes) if i not in cotxes_delete]

        frame = cv2.putText(frame, f"Cotxes que han pujat: {self.COTXES_PUGEN}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
        frame = cv2.putText(frame, f"Cotxes que han baixat: {self.COTXES_BAIXEN}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

        cv2.imshow("frame", frame)
        k = cv2.waitKey(1) & 0xFF

        


tracker = Tracker()


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
    if success and i%4 == 0:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.predict(frame, conf=0.5)[0]
        boxes = []
        if results.boxes.shape[0] !=0:
            for box in range(results.boxes.shape[0]):
                if results.boxes.cls[box] == 2:
                    boxes.append(results.boxes.xyxy[box,:].tolist())

        boxes = np.array(boxes)
        tracker.calculate_distances(boxes, frame)

    i+=1


print(boxes)