import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=2)
mpDraw = mp.solutions.drawing_utils


pTime = 0
cTime = 0
while True :
    success, img = cap.read()
    results = hands.process(img)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'fps:{int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2)
    cv2.imshow("Hands Tracking ", img)
    cv2.waitKey(1)