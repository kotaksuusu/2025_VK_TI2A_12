import cv2
from cvzone.PoseModule import PoseDetector


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
detector = PoseDetector()

while True:
    ok, img = cap.read()

    img = detector.findPose(img)
    lmlist, bboxInfo = detector.findPosition(img, draw=True, bboxWithHands=False)
    if lmlist :
        length, img, info = detector.findDistance(
            lmlist[11][0:2],
            lmlist[15][0:2],
            img=img,
            color=(255,0,0),
            scale = 10
        )
    cv2.imshow("image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

    

cap.release()
cv2.destroyAllWindows()