import cv2
import time
import mediapipe as mp
import numpy as np
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


class handDetectorPose():

    def __init__(self, static_image_mode=False,max_num_hands=1,model_complexity=1,  min_detection_confidence=0.5, min_tracking_confidence=0.5):

        self.static_image_mode  = static_image_mode
        self.max_num_hands = max_num_hands
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpDraw = mp.solutions.drawing_utils
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.static_image_mode,self.max_num_hands, self.model_complexity, self.min_detection_confidence,self.min_tracking_confidence)

    def find_hands(self,img,draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results =  self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def find_position(self,img, handNo=0,draw=True):
        lmList = []
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h,w,c,= img.shape
                #print(id, lm)
                cx,cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id,cx,cy])
                if draw:
                    cv2.circle(img, (cx,cy), 5, (255,0,0), cv2.FILLED)
        return lmList



def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = handDetectorPose(min_detection_confidence=0.7)


    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(
        IAudioEndpointVolume._iid_, CLSCTX_ALL, None)

    volume = cast(interface, POINTER(IAudioEndpointVolume))
    #volume.GetMasterVolumeLevel()
    volume_range = volume.GetVolumeRange()

    min_vol = volume_range[0]
    max_vol = volume_range[1]



    while True:
        success, img = cap.read()
        img = detector.find_hands(img)
        lmList = detector.find_position(img, draw=False)
        if len(lmList) !=0:
            #print(lmList[4], lmList[8])

            x1,y1 = lmList[4][1], lmList[4][2]
            x2,y2 = lmList[8][1], lmList[8][2]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(img, (x1,y1),7,(0,255,0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 7, (0, 255, 0), cv2.FILLED)
            cv2.line(img, (x1,y1), (x2,y2),(255,0,0), 3)
            cv2.circle(img, (cx, cy), 7, (0, 255, 0), cv2.FILLED)

            length = math.hypot(x2-x1,y2-y1)
            #print(length)

            vol = np.interp(length, [50,200], [min_vol,max_vol])
            print(int(length),vol)
            volume.SetMasterVolumeLevel(vol, None)




            if length < 100:
                cv2.line(img, (x1,y1), (x2,y2),(255,255,0), 3)
            if length < 50:
                mute = volume.GetMute()
                volume.SetMasterVolumeLevel(mute, None)


        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime


        cv2.putText(img, str(int(fps)),(70,50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == '__main__':
    main()


