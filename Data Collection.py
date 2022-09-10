#Note this project is subject to copyright

import tkinter as tk
import cv2
import os
import csv
import numpy as np
from tkinter import *
from PIL import Image, ImageTk


class Face_Recognition_System:
    def __init__(self, root):
        self.root = root
        self.root.geometry("1024x720+0+0")
        self.root.title("Final Year Project")

        img = Image.open(r"pictures\pic2.jpg")
        img = img.resize((130, 130), Image.ANTIALIAS)
        self.photoimg = ImageTk.PhotoImage(img)

        f_lbl = Label(self.root, image=self.photoimg)
        f_lbl.place(x=0, y=0, width=130, height=130)

        img1 = Image.open(r"pictures\pic1.jpg")
        img1 = img1.resize((1400, 130), Image.ANTIALIAS)
        self.photoimg1 = ImageTk.PhotoImage(img1)

        f_lbl = Label(self.root, image=self.photoimg1)
        f_lbl.place(x=130, y=0, width=1000, height=130)

        img2 = Image.open(r"pictures\pic2.jpg")
        img2= img2.resize((130, 130), Image.ANTIALIAS)
        self.photoimg2 = ImageTk.PhotoImage(img2)

        f_lbl = Label(self.root, image=self.photoimg2)
        f_lbl.place(x=1140, y=0, width=130, height=130)

        imgbg = Image.open(r"pictures\bg1.gif")
        imgbg = imgbg.resize((1530, 710), Image.ANTIALIAS)
        self.photoimgbg = ImageTk.PhotoImage(imgbg)

        bg_img = Label(self.root, image=self.photoimgbg)
        bg_img.place(x=0, y=130, width=1530, height=710)

        title_lbl = Label(bg_img, text="AUTOMATIC AI ATTENDANCE SYSTEM PROJECT",
                          font=("times new roman", 20, "bold"), bg="black", fg="green")
        title_lbl.place(x=200, y=0, width=800, height=53)

        lbl1 = Label(self.root, text="Registration Number", width=20, height=2, fg="black", bg="white",
                        font=('times', 15, ' bold '))

        lbl1.place(x=200, y=200)

        txt1 = Entry(self.root, width=20, bg="white", fg="black", font=('times', 15, ' bold '))
        txt1.place(x=550, y=215)

        lbl2 = Label(self.root, text="Enter Name", width=20 , fg="black", bg="white", height=2, font=('times', 15, ' bold '))
        lbl2.place(x=200, y=300)

        txt2 = Entry(self.root, width=20, bg="white", fg="black", font=('times', 15, ' bold ')  )
        txt2.place(x=550, y=315)

        lbl3 = Label(self.root, text="Notification →", width=20 , fg="black", bg="white", height=2, font=('times', 15, ' bold '))
        lbl3.place(x=200, y=400)

        message = Label(self.root, text="", bg="white", fg="black", width=30, height=2, font=('times', 15, ' bold '))
        message.place(x=550, y=400)

        def clearId():
            txt1.delete(0, 'end')

        def clearName():
            txt2.delete(0, 'end')

        def isNumber(s):
            try:
                float(s)
                return True
            except ValueError:
                pass

        def takeImages():
            name = (txt2.get())
            Id = (txt1.get())
            cap = cv2.VideoCapture(0)
            cap.set(3, 640)
            cap.set(4, 480)
            faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

            while True:
                ret, img = cap.read()
                # img = cv2.flip(img, -1)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                faces = faceCascade.detectMultiScale(gray, 1.3, 5)

                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    cv2.imwrite("SampleImages\ "+name +"."+Id  + ".jpg", gray[y:y + h, x:x + w])

                cv2.imshow('image', img)

                print('collected 1')



                k = cv2.waitKey(30)
                if k == 27:
                    break


            print("\n [INFO] Exiting Program and cleanup stuff")
            cap.release()
            cv2.destroyAllWindows()

            print("code ran successfully")



        def getImagesAndLabels(path):
            imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
            faces = []
            Ids = []
            for imagePath in imagePaths:
                pilImage = Image.open(imagePath).convert('L')
                imageNp = np.array(pilImage, 'uint8')
                Id = int(os.path.split(imagePath)[-1].split(".")[1])
                faces.append(imageNp)
                Ids.append(Id)
            return faces, Ids



        clearButton1 = Button(self.root, text="Clear", command=clearId, fg="black", bg="white", width=20, height=2, activebackground = "Red", font=('times', 15, ' bold '))
        clearButton1.place(x=850, y=200)

        clearButton2 = Button(self.root, text="Clear", command=clearName, fg="black", bg="white", width=20, height=2, activebackground = "Red", font=('times', 15, ' bold '))
        clearButton2.place(x=850, y=300)

        takeImg = Button(self.root, text="Take Images", command=takeImages, fg="black", bg="white", width=20, height=3, activebackground = "Green", font=('times', 15, ' bold '))
        takeImg.place(x=200, y=500)

        quitWindow = Button(self.root, text="Quit", command=root.destroy, fg="black", bg="white", width=20, height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
        quitWindow.place(x=500, y=500)

        lbl4 = Label(self.root, text="DESIGNED BY ADEBANJO MAYOWA SOLOMON. COVENANT UNIVERSITY, OTA.", width=80, fg="white", bg="black", font=('times', 15, ' bold'))
        lbl4.place(x=200, y=620)

if __name__ == "__main__":
    root = Tk()
    obj = Face_Recognition_System(root)
    root.mainloop()


