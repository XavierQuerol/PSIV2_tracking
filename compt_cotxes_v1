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
    
    
    
    def es_cotxe(self, coor_ant: list, coor:list, identificador: int ) -> bool:
        pass
    
    def afegir_cotxe(self,valor):
        
        if valor not in self._llista_cotxes :
            self._llista_cotxes.append(valor)
    
    def borrar_cotxe(self, valor):
        pass
    
    
@dataclass
class Cotxe:
    _x : int = 0
    _y : int = 0
    _w : int = 0
    _h : int = 0
    _id : int = 0
    _comptat: bool = False
    _centre: tuple = ()
    _puja : bool = False
    _baixa: bool = False
    
    def calcul_centre (self):
        self._centre= (self._x+self._w/2, self._y+self._h/2)
        return self._centre
    
    
    
    

def visualitza(img):
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()


cotxes = Cotxes()
cotxecitos = 0
cap = cv2.VideoCapture('seq/mini.mp4')
comptador_frame = 0
fgbg = cv2.createBackgroundSubtractorMOG2()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
while True:
    
    ret, frame = cap.read()
    if ret == False: break
    
    comptador_frame +=1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    

    #lines on mirar detecció

    cv2.line(frame, (0,int(frame.shape[0]/2)), (frame.shape[1],int(frame.shape[0]/2)) , (0,0,0), 4 )
    cv2.line(frame, (int(frame.shape[1]/2),0), (int(frame.shape[1]/2), int(frame.shape[0])) , (0,0,0), 4 )


    color = (0, 255, 0)
    if comptador_frame%1 == 0: 
        
        
        
        
        #area que miram
        area_pts = np.array([[200,340], [480,340], [620,frame.shape[0]], [50,frame.shape[0]]])
        

        imAux = np.zeros(shape=(frame.shape[:2]), dtype=np.uint8)
        imAux = cv2.drawContours(imAux, [area_pts], -1, (255), -1)
        image_area = cv2.bitwise_and(gray, gray, mask=imAux)

        # img binaria on regió blanc --> moviment
        fgmask = fgbg.apply(image_area)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        fgmask = cv2.dilate(fgmask, (3,3), iterations=5)
        
        
        # Trobar contorns 
        cnts = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        for cnt in cnts:
            if cv2.contourArea(cnt) > 3500:
                
                x, y, w, h = cv2.boundingRect(cnt)
                cotxe = Cotxe(x,y,w,h)
                x_centre , y_centre = cotxe.calcul_centre()
                cv2.rectangle(frame, (x,y), (x+w, y+h),(0,255,0), 2)
                color = (0, 0, 255)    
                
                
                
                
                if cotxe._comptat == False and x!=0:
                    
              
                    if int(frame.shape[0]/2) +3 > y_centre >int(frame.shape[0]/2) -3 :
                        if len(cotxes._llista_cotxes) ==0:
                            cotxe._comptat == True
                            cotxe._id = comptador_frame
                            cotxecitos +=1
                            cotxes.afegir_cotxe(cotxe)
                        

                            if x_centre < int(frame.shape[1]/2):
                                cotxe._baixa = True
                            else:
                                cotxe._puja = True
                        else: 
                            
                            if  abs ( cotxes._llista_cotxes[-1]._id - comptador_frame ) > 25:
                                    cotxe._comptat == True
                                    cotxe._id = comptador_frame
                                    cotxecitos +=1
                                    cotxes.afegir_cotxe(cotxe)
                                  #  print(x_centre , int(frame.shape[1]/2))
                                    if x_centre < int(frame.shape[1]/2)-15:
                                        cotxe._baixa = True
                                    else:
                                        cotxe._puja = True
                      
               
                
               
                cv2.drawContours(frame, [area_pts], -1, color, 2)
            
                cv2.imshow('fgmask', fgmask)
                cv2.imshow("frame", frame)
                k = cv2.waitKey(70) & 0xFF
                if k == 27:
                    break

cap.release()
cv2.destroyAllWindows()



print("Cotxes trobats:", cotxecitos)
puja = 0
baixa = 0
for i in cotxes._llista_cotxes:
    if i._puja == True:
        puja+=1
    else:
        baixa+=1
        
print("Cotxes que pugen:", puja)
print("Cotxes que baixen", baixa)




