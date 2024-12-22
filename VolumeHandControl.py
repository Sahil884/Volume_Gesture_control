import cv2
import time

import numpy as np

import HandTrackingModule as htm
import numpy
import math
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

wCam, hCam = 640, 480


cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
ptime = 0

detector = htm.handDetector(detectionCon=0.7)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
        IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()

minVol = volRange[0]
maxVol = volRange[1]
vol_bar = 400
vol_percent = 0

while True:
    istrue, frame = cap.read()
    frame = detector.findHands(frame)
    lmList = detector.findPosition(frame, draw=False)
    if len(lmList) != 0:
        # print(lmList[4], lmList[8])
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1+x2)//2, (y1+y2)//2  # center x and center y

        cv2.circle(frame, (x1, y1), 15, (255,0,255), cv2.FILLED)
        cv2.circle(frame, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
        cv2.line(frame, (x1,y1), (x2,y2), (255,0,255), 3) # line between 2 point of the hand
        cv2.circle(frame, (cx,cy), 15, (255,0,255), cv2.FILLED) # center of the line

        length = math.hypot(x2-x1, y2-y1)
        # print(length)

        # Hand Range 28 - 280
        # volume range -96 - 0

        vol = np.interp(length, [28, 270], [minVol, maxVol]) # converts length to volume
        vol_bar = np.interp(length, [28, 270], [400, 150])
        vol_percent = np.interp(length, [28,270], [0,100])

        if length < 50:
            cv2.circle(frame, (cx,cy), 15, (0,255,0), cv2.FILLED)

        volume.SetMasterVolumeLevel(vol, None)

    cv2.rectangle(frame, (50, 150), (85,400), (0,255,0), 3)
    cv2.rectangle(frame, (50, int(vol_bar)), (85, 400), (0, 255, 0), cv2.FILLED)

    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime

    cv2.putText(frame, f'FPS: {int(fps)}', (40,70), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,255), 3)
    cv2.putText(frame, f' {int(vol_percent)}%', (30, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


