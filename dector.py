import cv2
import numpy as np

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)
rec = cv2.face.LBPHFaceRecognizer_create();
rec.read('trainner\\trainner.yml')
id = 0
font = cv2.FONT_HERSHEY_COMPLEX
while (True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (130, 212, 200), 2)
        id,cinf=rec.predict(gray[y:y+h,x:x+w])
        if(id==0):
            id = "Unknown"
        elif(id==1):
            id = "Abhishek"
        elif(id==3):
            id = "Aditya"
        elif(id==2):
            id="Saurabh"
        else:
            id = "Unknown"
        cv2.putText(img,str(id),(x,y+h),font,1,(10,0,0),2);
        
    cv2.imshow("Face Detector", img)
    if (cv2.waitKey(1) == ord('q')):
        break

cam.release()
cv2.destroyAllWindows()
