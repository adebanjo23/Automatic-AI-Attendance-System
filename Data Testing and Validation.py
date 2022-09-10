#Note this project is subject to copyright

import cv2
import numpy as np
import os
from datetime import datetime

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml5')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

font = cv2.FONT_HERSHEY_SIMPLEX

#indicate id counter




cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4,480)

minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDatalist = f.readlines()
        nameList = []
        for line in myDatalist:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')



while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH))

    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img,(x,y),(x+w, y+h),(0,255,0),2)

        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

        if (confidence > 20):
            id = str(id)

            cv2.putText(img,id, (h+6,w-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendance(id)


        else:
            id = 'unknown'
            cv2.putText(img,id, (h+6,w-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)



    cv2.imshow('camera', img)
    k = cv2.waitKey(1)
    if k == 27:
        break




cam.release()
cv2.destroyAllWindows()