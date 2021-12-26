import cv2
import mediapipe as mp
import time
import poseModule as pm


cap = cv2.VideoCapture(0)
pTime = 0


mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(0.75)


while True:
    success,img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    print(results)

    if results.detections:
        for id, detection in enumerate(results.detections):

            print(detection.location_data.relative_bounding_box)
            bboxC = detection.location_data.relative_bounding_box
            h, w, c, = img.shape

            bboxes = []
            bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), \
                    int(bboxC.width * w), int(bboxC.height * h)
            bboxes.append([id,bbox, detection.score])
            cv2.rectangle(img,bbox, (255,0,0), 2)
            cv2.putText(img, f'{int(detection.score[0] * 100)} %', (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2,(0,255,0),2)
            print("Certainty:", detection.score[0])




    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime=cTime

    cv2.putText(img, str(int(fps)),(70,50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
