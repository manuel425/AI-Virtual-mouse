import cv2
import mediapipe as mp
import time
import math


class HandDetector():
    def __init__(self, mode=False, MaxHands=1, MinDetectionConfidence=0.5, MinTrackingConfidence=0.5):
        self.mode = mode
        self.MaxHands = MaxHands
        self.MinDetectionConfidence = MinDetectionConfidence
        self.MinTrackingConfidence = MinTrackingConfidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.MaxHands, self.MinDetectionConfidence,
                                        self.MinTrackingConfidence)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def FindHands(self, img, draw=True):

        RGBimage = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.result = self.hands.process(RGBimage)
        # This prints the multihand landmark
        # print(result.multi_hand_landmarks)

        if self.result.multi_hand_landmarks:
            for handLMS in self.result.multi_hand_landmarks:
                # for id, LM in enumerate(handLMS.landmark):
                #     print(id, LM)
                #     h, w, c = img.shape
                #     cx, cy = int(LM.x * w), int(LM.y * h)
                #     print(id, cx, cy)
                #     if id == 8:
                #         cv2.circle(img, (cx, cy), 15, (255, 255, 0, cv2.FILLED))
                if draw:
                    self.mpDraw.draw_landmarks(img, handLMS, self.mpHands.HAND_CONNECTIONS)

        return img

    # RGBimage = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # result = hands.process(RGBimage)
    # This prints the multihand landmark
    # print(result.multi_hand_landmarks)

    def FindPosition(self, img, HandNo=0, draw=False):
        xList = []
        yList = []
        bbox = []

        self.LMList = []
        if self.result.multi_hand_landmarks:
            MyHand = self.result.multi_hand_landmarks[HandNo]
            for id, LM in enumerate(MyHand.landmark):
                # print(id, LM)
                h, w, c = img.shape
                cx, cy = int(LM.x * w), int(LM.y * h)
                xList.append(cx)
                yList.append(cy)
                # print(id, cx, cy)
                self.LMList.append([id, cx, cy])

                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 255, 0, cv2.FILLED))
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(img, (xmin - 10, ymin - 10), (xmax + 10, ymax + 10), (0, 255, 0), 4)

        return self.LMList, bbox

    def FingersUp(self):
        fingers = []
        
        if self.LMList[self.tipIds[0]][1] < self.LMList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        
        for id in range(1, 5):
            if self.LMList[self.tipIds[id]][2] < self.LMList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

    def DetermineDistance(self, p1, p2, img, draw=True, r=15, t=3):
        x1, y1 = self.LMList[p1][1:]
        x2, y2 = self.LMList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (255, 0, 255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]


def main():
    PrevTime = 0
    CurrentTime = 0
    Capture = cv2.VideoCapture(0)
    Detector = HandDetector()
    while True:
        success, img = Capture.read()
        img = Detector.FindHands(img)
        LMList = Detector.FindPosition(img)
        if len(LMList) != 0:
            print(LMList[4])

        CurrentTime = time.time()
        fps = 1 / (CurrentTime - PrevTime)
        PrevTime = CurrentTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 3, (0, 255, 255, 3))

        cv2.imshow("Image", img)
        cv2.waitKey(3)


if __name__ == "__main__":
    main()
