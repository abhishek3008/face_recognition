import cv2
import numpy as np

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)

id=raw_input('Enter the user id')
samplenum=0;
while (True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        samplenum=samplenum+1;
        cv2.imwrite("dataSet/User."+str(id)+"."+str(samplenum)+".jpg",gray[y:y+h,x:x+w])
        cv2.rectangle(img, (x, y), (x + w, y + h), (130, 212, 200), 2)
    cv2.imshow("Face Detector", img)
    cv2.waitKey(1);
    if(samplenum>10):
        break

cam.release()
cv2.destroyAllWindows()
exit(0)
