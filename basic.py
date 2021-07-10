import cv2
import mediapipe as mp
import time 

capture = cv2.VideoCapture(0, )

mpHands = mp.solutions.hands
hands = mpHands.Hands()

mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    success, img = capture.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)

    h, w, c = img.shape

    if results.multi_hand_landmarks:
        for handLandMrks in results.multi_hand_landmarks:
            for id, landm in enumerate(handLandMrks.landmark):
                cx, cy = int(landm.x*w), int(landm.y*h)
                print(id, cx, cy)
                if id == 4:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLandMrks, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 4)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

