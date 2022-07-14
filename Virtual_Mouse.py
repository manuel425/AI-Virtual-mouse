import cv2
import numpy as np
import CodeModule as CM
import time
import autopy

####################
wCam, hCam = 640,480
frameR = 100 #Frame Reduction
SmootheningValue = 5
####################

PrevTime=0
PrevLocationX, PrevLocationY = 0,0
CurrLocationX, CurrLocationY = 0,0



cap = cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)

detector = htm.HandDetector(MaxHands=1)

wScreen, hScreen = autopy.screen.size()
print(wScreen,hScreen)

while True:   
    success, img = cap.read()
    img = detector.FindHands(img)
    LMList, bbox = detector.FindPosition(img, draw=True)

    
    if len(LMList)!=0:
        x1,y1 = LMList[8][1:]
        x2,y2 = LMList[12][1:]

        #print(x1,y1,x2,y2)

        
        FingersRaised = detector.FingersUp()        
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR),
        (255, 0, 255), 3)
        
        if FingersRaised[1]==1 and FingersRaised[2]==0:

            x3 = np.interp(x1, (frameR,wCam-frameR), (0,wScreen))
            y3 = np.interp(y1, (frameR, hCam-frameR), (0,hScreen))
            
            CurrLocationX = PrevLocationX + (x3 - PrevLocationX)/5
            CurrLocation = PrevLocationY + (y3 - PrevLocationY) / 5
            
            autopy.mouse.move(wScreen-CurrLocationX, CurrLocationY)
            cv2.circle(img,(x1,y1),15,(255,0,255), cv2.FILLED)
            PrevLocationX, PrevLocationY = CurrLocationX, CurrLocationY


        
        if FingersRaised[1]==1 and FingersRaised[2]==1:           
            length, img, LineInfo = detector.DetermineDistance(8, 12, img)
            print(length)
            
            if length < 30:
                cv2.circle(img, (LineInfo[4], LineInfo[5]), 15, (255, 255, 0), cv2.FILLED)
                autopy.mouse.click()




    CurrTime = time.time()
    Framerate = 1/(CurrTime-PrevTime)
    PrevTime = CurrTime
    cv2.putText(img,str(int(Framerate)), (20,50), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 3, (255,0,0), 3)


    cv2.imshow("manuel's mouse panel", img)
    cv2.waitKey(1)
