import numpy as np
import cv2

class Cotxe():
    def __init__(self, x, y, x2, y2, id):
        self.cornerTL = [x,y]
        self.cornerBR = [x2,y2]
        self.id = id
        self.frames = 0
        self.show = True
        self.up = False
        self.down = False
        self.TROBAT = False
        self.trajec = [(x,y)]



class Tracker():

    def __init__(self, metode_comptar):
        self.cotxes = []
        self.nou_id = 0
        self.COTXES_PUGEN = 0
        self.COTXES_BAIXEN = 0
        self.metode_comptar = metode_comptar


    def recalculate(self, bboxes, frame):
        if len(bboxes)>0:
            if len(self.cotxes) == 0:
                for i in range(bboxes.shape[0]):
                    self.nou_id += 1
                    id = self.nou_id
                    cotxe = Cotxe(x=bboxes[i,0], y=bboxes[i,1], x2=bboxes[i,2], y2=bboxes[i,3], id=id)
                    self.cotxes.append(cotxe)
            else:
                for cotxe in self.cotxes:
                    cotxe.show = False
                
                distances = self.calculate_distances(bboxes)
                self.update_cars(distances, bboxes)

            cotxes_delete = []       
            for i, cotxe in enumerate(self.cotxes):
                if cotxe.show==True:
                    cotxe.frames = 0
                    frame = self.paint_id(cotxe, frame)
                    if self.metode_comptar == "one_line":
                        self.one_line(cotxe)
                    elif self.metode_comptar == "two_lines":
                        self.two_lines(cotxe)
                else:
                    cotxe.frames += 1
                    if cotxe.frames >15:
                        if self.metode_comptar == "c_final":
                            self.c_final(cotxe)
                        cotxes_delete.append(i)

            self.cotxes = [cotxe for i, cotxe in enumerate(self.cotxes) if i not in cotxes_delete]

        if self.metode_comptar == "one_line":
            frame = cv2.line(frame, (0,600), (539,600),(0,255,255), 10)
        elif self.metode_comptar == "two_lines":
            frame = cv2.line(frame, (0,550), (539,550), (0,255,255), 10)
            frame = cv2.line(frame, (0,700), (539,700), (0,255,255), 10)
        frame = cv2.putText(frame, f"Cotxes que han pujat: {self.COTXES_PUGEN}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
        frame = cv2.putText(frame, f"Cotxes que han baixat: {self.COTXES_BAIXEN}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

        cv2.imshow("frame", frame)
        k = cv2.waitKey(1) & 0xFF

        return frame

    def calculate_distances(self, bboxes):
        
        cotxes_actuals = np.zeros((len(self.cotxes), 2))
        cotxes_nous = bboxes[:,:2]
        for i in range(len(self.cotxes)):
            cotxes_actuals[i,:] = self.cotxes[i].cornerTL


        distances = np.zeros((cotxes_nous.shape[0], cotxes_actuals.shape[0]))

        for i in range(cotxes_nous.shape[0]):
            for j in range(cotxes_actuals.shape[0]):
                distances[i,j] = np.linalg.norm(cotxes_nous[i,:]-cotxes_actuals[j,:])
        
        return distances
    
    def update_cars(self, distances, bboxes):

        minims = distances.argmin(axis=0)
        minims2 = distances.argmin(axis=1)

        for i in range(len(minims2)):
            j = minims2[i]
            if minims[j] == i and distances[i, j] < 100:
                self.cotxes[j].cornerTL = bboxes[i,:2]
                self.cotxes[j].cornerBR = bboxes[i,2:]
                self.cotxes[j].show = True
                self.cotxes[j].trajec.append((bboxes[i,:2][0], bboxes[i,:2][1]))
            else:
                self.nou_id += 1
                id = self.nou_id
                cotxe = Cotxe(x=bboxes[i,0], y=bboxes[i,1], x2=bboxes[i,2], y2=bboxes[i,3], id=id)
                self.cotxes.append(cotxe)
        
    def paint_id(self, cotxe, frame):
        frame = cv2.putText(frame, str(cotxe.id), (int(cotxe.cornerTL[0]), int(cotxe.cornerTL[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 255, 2)
        frame = cv2.rectangle(frame, (int(cotxe.cornerTL[0]),int(cotxe.cornerTL[1])), (int(cotxe.cornerBR[0]), int(cotxe.cornerBR[1])), 255, 5)
        return frame
                    
    def c_final(self, cotxe):
        if cotxe.trajec[0][1] - cotxe.trajec[-1][1] > 300:
            self.COTXES_PUGEN += 1
        elif cotxe.trajec[0][1] - cotxe.trajec[-1][1] < -300:
            self.COTXES_BAIXEN += 1

    def one_line(self, cotxe):
        if not cotxe.TROBAT:
            if cotxe.cornerTL[1]<620:
                cotxe.up = True
                if cotxe.up*cotxe.down:
                    self.COTXES_PUGEN +=1
                    cotxe.TROBAT = True
            else:
                cotxe.down = True
                if cotxe.up*cotxe.down:
                    self.COTXES_BAIXEN +=1
                    cotxe.TROBAT = True
    
    def two_lines(self, cotxe):
        if not cotxe.TROBAT:
            if cotxe.cornerTL[1]<550:
                cotxe.up = True
                if cotxe.up*cotxe.down:
                    self.COTXES_PUGEN +=1
                    cotxe.TROBAT = True
            elif cotxe.cornerTL[1]>700:
                cotxe.down = True
                if cotxe.up*cotxe.down:
                    self.COTXES_BAIXEN +=1
                    cotxe.TROBAT = True