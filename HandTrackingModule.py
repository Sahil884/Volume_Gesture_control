import cv2
import mediapipe as mp
import time
import math


class handDetector():
    def __init__(self,mode=False, maxHands = 2, detectionCon=0.5, trackCon=0.5, complexity=1):
        self.mode = mode
        self.maxHands = maxHands
        self.complexity = complexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.complexity,
                                        self.detectionCon, self.trackCon)  # initializing mediapipe hand object
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, frame, draw=True):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(frame_rgb)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame, handLms,
                                               self.mpHands.HAND_CONNECTIONS)

        return frame


    def findPosition(self, frame, handNo=0, draw=True):

        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):  # this will give id and landmark coordinates of all 21 landmarks
                # print(id,lm)
                height, width, channels = frame.shape
                cx, cy = int(lm.x * width), int(lm.y * height)
                lmList.append([id, cx, cy])
                # print(id, cx, cy)
                if draw:
                    cv2.circle(frame, (cx, cy), 7, (255, 0, 255), cv2.FILLED)

        return lmList


def main():
    # for fps
    previous_time = 0
    current_time = 0
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    detector = handDetector()

    while True:
        istrue, frame = cap.read()
        frame = detector.findHands(frame)
        lmList = detector.findPosition(frame, draw=False)
        if len(lmList) != 0:
            print(lmList[4])

        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time

        cv2.putText(frame, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 255), 3)

        cv2.imshow('video', frame)

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
