import cv2
import numpy as np

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)


id = raw_input('enter user id')
sampleNumber = 0

while(True):
    ret,img = cam.read();
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)
    for(x, y, w, h) in faces:
        sampleNumber = sampleNumber + 1
        cv2.imwrite("dataSet/user."+str(id)+"."+str(sampleNumber)+".jpg", gray[y:y+h, x:x+w]) 
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)
        cv2.waitKey(100)
    cv2.imshow("Face", img)
    cv2.waitKey(1)
    if(sampleNumber>20):
        break
cam.release()
cv2.destroyAllWindows()
    
