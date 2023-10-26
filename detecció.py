from dataclasses import dataclass, field
from typing import *

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

class Cotxe():
    def __init__(self, id):
        self.id = id
        self.trajec= []
        #self.cent = None
        self.sentit = None # 0 --> puja, 1 --> baixa

class Repte2():
    def __init__(self):
        self.cotxes = {} # self.cotxes[id] = objecte cotxe amb quest id
        self.anterior = [] #[(centre, id)]


    def delimitar_zona(self, frame):
        area_pts = np.array([[200,340], [480,340], [620,frame.shape[0]], [50,frame.shape[0]]])
        imAux = np.zeros(shape=(frame.shape[:2]), dtype=np.uint8)
        imAux = cv2.drawContours(imAux, [area_pts], -1, (255), -1)
        image_area = cv2.bitwise_and(frame, frame, mask=imAux)
        return image_area
    

    def prepro(self, frame, fgbg):

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(3,2)) #horitzontal
        kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(1,3)) #veritcal  

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(gray, (11, 11), 0)

        fgmask = fgbg.apply(img)

        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        fgmask = cv2.dilate(fgmask, kernel2, iterations=15)
        #fgmask = cv2.erode(fgmask, kernel2, iterations=10)
        fgmask = cv2.erode(fgmask, kernel3, iterations=13)
        fgmask = cv2.dilate(fgmask, kernel3, iterations=7)

        #fgmask[fgmask != 255] = 0

        return fgmask

    def dist_cent(self, centre1, centre2):
        x1, y1 = centre1
        x2, y2 = centre2
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return distance

    def trobar_cotxes(self, img, frame, i):

        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        cnts = []

        if len(contours) != 0 :
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                nou_centre = (x+w/2, y+h/2)
                    

                if cv2.contourArea(cnt) > 7000:
                    cv2.rectangle(frame, (x,y), (x+w, y+h), 255, 5)
                    
                    cnts.append(cnt)

                    if len(self.cotxes.keys()) == 0:
                        c = Cotxe(0)
                        c.trajec.append(nou_centre)
                        self.cotxes[0] = c
                        self.anterior.append((nou_centre, 0))
                        cv2.putText(frame, str(0), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 255, 2)

                    else:
                        for (centre,id_a) in self.anterior:
                            self.anterior = []
                            if abs(self.dist_cent(nou_centre,centre)) < 180: 
                                self.cotxes[id_a].trajec.append(nou_centre)
                                self.anterior.append((nou_centre, id_a))
                                cv2.putText(frame, str(id_a), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 255, 2)

                                
                            else:
                                id = max(self.cotxes.keys()) + 1
                                c = Cotxe(id)
                                c.trajec.append(nou_centre)
                                self.cotxes[id] = c
                                self.anterior.append((nou_centre, c.id))
                                cv2.putText(frame, str(id), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 255, 2)

    def sentit(self, cotxe):
        if len(cotxe.trajec) > 25: # ha trobat cotxe en minim 5 frames
            if cotxe.trajec[0][1] < cotxe.trajec[-1][1]:
                cotxe.sentit = 1
            else:
                cotxe.sentit = 0




                




def main(path): #main
    fgbg = cv2.createBackgroundSubtractorKNN()
    videoPath = path
    cap = cv2.VideoCapture(videoPath)
    ret = True
    #crear classe detector
    rep = Repte2()
    i = 0
    p = 0
    b = 0
    while ret:
        ret, frame = cap.read()
        if ret:
            i+=1
            #crida delimitar zona 
            img = rep.delimitar_zona(frame)
            #crida prepro 
            img = rep.prepro(img, fgbg)
            #crida trobar cotxes 
            rep.trobar_cotxes(img, frame, i)
            cv2.drawContours(frame, [np.array([[200,340], [480,340], [620,frame.shape[0]], [50,frame.shape[0]]])], -1, (0, 255, 0), 2)

            cv2.imshow('fgmask', img)
            cv2.imshow("frame", frame)
            k = cv2.waitKey(1) & 0xFF
    #crida comptar cotxes
    for c in rep.cotxes.values():
        rep.sentit(c)
        if c.sentit ==0:
            p += 1
        elif c.sentit ==1:
            b +=1
    print(f'Cotxes que pugen: {p}\nCotxes que baixen: {b}\nTotal: {p+b}')

    

videoPath = '/Users/abriil/Uni/23-24/PSIV2/repo_tracking/output7.mp4'
main(videoPath)
