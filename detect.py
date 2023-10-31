from dataclasses import dataclass, field
from typing import *

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.ndimage import binary_fill_holes

class Detect():

    def __init__(self, w = 960, h = 540):
        self.shape = [w,h]
        self.area_pts = np.array([[200,360], [410,360], [h-35,w], [50,w]])
        self.min_area = 6000
    
    def delimitar_zona(self, frame):
        
        imAux = np.zeros(shape=(frame.shape[:2]), dtype=np.uint8)
        imAux = cv2.drawContours(imAux, [self.area_pts], -1, (255), -1)
        image_area = cv2.bitwise_and(frame, frame, mask=imAux)
        cv2.drawContours(frame, [self.area_pts], -1, (0, 255, 0), 2)
        return image_area
    
    def prepro(self, frame, fgbg):

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(3,1)) #horitzontal
        kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(1,5)) #veritcal  

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(gray, (11, 11), 0)

        fgmask = fgbg.apply(img)
        fgmask = cv2.dilate(fgmask, kernel2, iterations=9)
        fgmask = cv2.erode(fgmask, kernel, iterations=4)
        fgmask = cv2.dilate(fgmask, kernel3, iterations=4)
        #fgmask[fgmask != 127] = 0
        #fgmask[fgmask == 127] = 255
        #fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        #fgmask = cv2.dilate(fgmask, kernel, iterations=3)
        #fgmask = cv2.dilate(fgmask, kernel3, iterations=7)
        #fgmask = binary_fill_holes(fgmask)*255
        #fgmask = cv2.erode(fgmask, kernel, iterations=1)
        #fgmask = cv2.erode(fgmask, kernel, iterations=3)
        #fgmask = cv2.dilate(fgmask, kernel2, iterations=7)
        #fgmask = cv2.dilate(fgmask, kernel3, iterations=11)
        #fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
        fgmask = fgmask.astype(np.uint8)

        return fgmask
    
    def trobar_contorns(self, img):
        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        cnts = []

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if cv2.contourArea(cnt) > self.min_area and h<1.5*w: #and w>h:
                cnts.append([x,y,w,h])
        
        return cnts