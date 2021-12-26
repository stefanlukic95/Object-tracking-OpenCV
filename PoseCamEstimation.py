import cv2
import mediapipe as mp
import time


cap = cv2.VideoCapture(0)
pTime = 0


mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDrawH = mp.solutions.drawing_utils


while True:
    (success, frame) = cap.read()

    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    resultsHands = hands.process(imgRGB)

    if results.pose_landmarks:
        mpDraw.draw_landmarks(frame, results.pose_landmarks,mpPose.POSE_CONNECTIONS)


    if resultsHands.multi_hand_landmarks:
        for handLms in resultsHands.multi_hand_landmarks:

            for id1, lm1 in enumerate(handLms.landmark):
                h1, w1, c1, = frame.shape
                print(id1, lm1)
                cx1, cy1 = int(lm1.x * w1), int(lm1.y * h1)
                cv2.circle(frame, (cx1, cy1), 5, (0, 0, 255), cv2.FILLED)
            mpDrawH.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)




    for id, lm in enumerate(results.pose_landmarks.landmark):
        h, w, c, = frame.shape
        print(id, lm)
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), cv2.FILLED)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime=cTime

    cv2.putText(frame, str(int(fps)),(70,50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)
    cv2.imshow("Image", frame)
    key = cv2.waitKey(10)

    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()