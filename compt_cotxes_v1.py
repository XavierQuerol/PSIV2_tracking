# -*- coding: utf-8 -*-

from dataclasses import dataclass, field
from typing import *

import cv2
import numpy as np
import imutils
import matplotlib.pyplot as plt
import time



@dataclass
class Cotxes :
    _llista_cotxes : list = field(default_factory=list)
    _comptador_cotxes : int = 0
    
    def delimitar_zona(self, frame):
        area_pts = np.array([[200,340], [480,340], [620,frame.shape[0]], [50,frame.shape[0]]])
        imAux = np.zeros(shape=(frame.shape[:2]), dtype=np.uint8)
        imAux = cv2.drawContours(imAux, [area_pts], -1, (255), -1)
        image_area = cv2.bitwise_and(frame, frame, mask=imAux)
        return image_area
    
    def es_cotxe(self, coor_ant: list, coor:list, identificador: int ) -> bool:
        pass
    
    def afegir_cotxe(self,valor):
        if valor not in self._llista_cotxes :
            self._llista_cotxes.append(valor)
    
    def preprocess(self,img,fgbg):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(8,3)) #horitzontal
        kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
        fgmask = fgbg.apply(img)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        fgmask = cv2.dilate(fgmask, (10,10), iterations=5)
        return fgmask
    
    
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
        fgmask = fgmask.astype(np.uint8)

        return fgmask
    
    def trobar_cotxes(self, frame, fgmask, comptador_frame):
        cnts = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        for cnt in cnts:
            x, y, w, h = cv2.boundingRect(cnt)
            if cv2.contourArea(cnt) > 3500 and (w<1.4*h and w>0.8*h):
    
    
                cotxe = Cotxe(x,y,w,h)
                x_centre , y_centre = cotxe.calcul_centre()
                cotxe.definir_sentit(x_centre,frame.shape[1])
                cv2.rectangle(frame, (x,y), (x+w, y+h),(0,255,0), 2)
                color = (0, 0, 255)    
                cotxe._id = comptador_frame

            
                if int(frame.shape[0]/2) +20 > y_centre >int(frame.shape[0]/2) -20:
                        if len(self._llista_cotxes) ==0:
                            cotxe._comptat = True
                            self._comptador_cotxes +=1
                            self.afegir_cotxe(cotxe)
                          
                        elif  abs ( self._llista_cotxes[-1]._id -  cotxe._id  ) > 20:
                                  cotxe._comptat = True
                                  self._comptador_cotxes +=1
                                  self.afegir_cotxe(cotxe)
    
                        elif len(self._llista_cotxes) >=2 and \
                                abs ( self._llista_cotxes[-2]._id -  cotxe._id  ) <20 \
                                    and self._llista_cotxes[-2]._sentit!= cotxe._sentit:                          
                              cotxe._comptat = True
                              self._comptador_cotxes +=1
                              self.afegir_cotxe(cotxe)
                  
               
    def comptar_cotxes(self):
        
        print("Cotxes trobats:", self._comptador_cotxes)
        puja = 0
        baixa = 0
        for i in self._llista_cotxes:
            if i._sentit == True:
                puja+=1
            else:
                baixa+=1
                
        print("Cotxes que pugen:", puja)
        print("Cotxes que baixen", baixa)


    
@dataclass
class Cotxe:
    _x : int = 0
    _y : int = 0
    _w : int = 0
    _h : int = 0
    _id : int = 0
    _comptat: bool = False
    _centre: tuple = ()
    _sentit : bool = None
    
    
   
    
    def calcul_centre (self):
        self._centre= (self._x+self._w/2, self._y+self._h/2)
        return self._centre
    
    
    def definir_sentit(self, x_centre,shape):
        if x_centre < int(shape/2)-20:
            self._sentit = False #baixa
        else:
            self._sentit = True #puja
    






def main(path):
    cotxes = Cotxes()
    cap = cv2.VideoCapture(path)
    comptador_frame = 0
    fgbg = cv2.createBackgroundSubtractorMOG2()
    while True:
        
        ret, frame = cap.read()
        if ret == False: break
        
        comptador_frame +=1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.line(frame, (0,int(frame.shape[0]/2)), (frame.shape[1],int(frame.shape[0]/2)) , (0,0,0), 4 )
     #   cv2.line(frame, (int(frame.shape[1]/2),0), (int(frame.shape[1]/2), int(frame.shape[0])) , (0,0,0), 4 )
        color = (0, 255, 0)
            
        image_area = cotxes.delimitar_zona( frame)   
        #fgmask = cotxes.preprocess(image_area,fgbg)
        fgmask = cotxes.prepro(image_area, fgbg)
        cotxes.trobar_cotxes(frame, fgmask, comptador_frame)
        
                    
        cv2.drawContours(frame, [np.array([[200,340], [480,340], [620,frame.shape[0]], [50,frame.shape[0]]])], -1, color, 2)
        cv2.imshow('fgmask', fgmask)
        cv2.imshow("frame", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    cotxes.comptar_cotxes()


path = 'seq/output2.mp4'
main(path)
