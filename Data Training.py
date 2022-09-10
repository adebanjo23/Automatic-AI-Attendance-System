#Note this project is subject to copyright

#importing the required libraries/modules
import numpy as np
import cv2
from PIL import Image
import os




path = 'SampleImages'

recognizer = cv2.face.LBPHFaceRecognizer_create()

detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def getImagesAndlabels(path):

    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    ids = []

    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L')
        img_numpy = np.array(PIL_img, 'uint8')

        id = os.path.split(imagePath)[-1].split(".")[1]
        faces = detector.detectMultiScale(img_numpy)

        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y+h, x:x+w])
            id = int(id)
            ids.append(id)

    return faceSamples,ids

print("\n [INFO] Training faces it will take a few seconds. wait ")

faces,ids, = getImagesAndlabels(path)

recognizer.train(faces, np.array(ids))


recognizer.write('trainer/trainer.yml5')

print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))



print("code ran successfully")