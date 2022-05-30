import numpy as np
import cv2
import os
import h5py
import dlib
from imutils import face_utils
from tensorflow.keras.models import load_model
import sys
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.layers import Dense, Activation, Flatten
from PIL import Image
from Model import model as md
from scripts.marker import mark
import csv
from datetime import date
from datetime import datetime

labels = []

srcpath = ''


def markAttendance():

    return


def getImagesAndLabels():

    path = os.path.join(srcpath, 'dataset')
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    ids = []

    for imagePath in imagePaths:

        # in case of error
        try:
            PIL_img = Image.open(imagePath).convert(
                'L')  # convert it to grayscale
        except:
            continue
        img_numpy = np.array(PIL_img, 'uint8')

        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faceSamples.append(img_numpy)
        ids.append(id)
        csvPath = os.path.join(srcpath, 'list/idname.csv')
        with open(csvPath, mode='r') as file:
            csvFile = csv.reader(file)
            for lines in csvFile:
                labels.append(lines[0])
    return faceSamples, ids


def start(srcPath):
    PrsList = []
    strtme = datetime.now().strftime("%H:%M:%S")
    srcpath = srcPath
    _, ids = getImagesAndLabels()
    model = md((32, 32, 1), len(set(ids)))
    model.load_weights(os.path.join(srcpath, 'trainedModel/trained_model.h5'))
    model.summary()
    cascPath = os.path.join(
        srcpath, "haarcascade/haarcascade_frontalface_default.xml")
    faceCascade = cv2.CascadeClassifier(cascPath)
    font = cv2.FONT_HERSHEY_COMPLEX
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)  # set video widht
    cap.set(4, 480)  # set video height
    print('Video Capture Started')
    ret = True

    clip = []
    while ret:
        # read frame by frame
        ret, frame = cap.read()
        nframe = frame
        faces = faceCascade.detectMultiScale(
            frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30))

        try:
            (x, y, w, h) = faces[0]
        except:
            continue
        frame = frame[y:y+h, x:x+w]
        frame = cv2.resize(frame, (32, 32))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('result small', frame)
        c = cv2.waitKey(1)
        if c & 0xFF == ord('q'):
            break

        gray = gray[np.newaxis, :, :, np.newaxis]
        #gray = gray.reshape(-1, 32, 32, 1).astype('float32') / 255.
        print(gray.shape)
        prediction = model.predict(gray)
        print("prediction:" + str(prediction))
        print("\n\n\n\n")
        print("----------------------------------------------")

        prediction = prediction.tolist()
        listv = prediction[0]
        n = listv.index(max(listv))
        print("\n")
        print("----------------------------------------------")
        print("Highest Probability: " +
              labels[n] + " ==> " + str(prediction[0][n]))
        print("Highest Probability: " + "User " +
              str(n) + " ==> " + str(prediction[0][n]))

        print("----------------------------------------------")
        print("\n")
        for (x, y, w, h) in faces:
            try:
                confidence = prediction[0][n]
                cv2.rectangle(nframe, (x, y), (x+w, y+h), (0, 255, 0), 2)
                id = "H"
                conf = "0"
                if (confidence <= 100):
                    id = labels[n]
                    conf = "  {0}%".format(round(100 - confidence))
                else:
                    id = "unknown"
                    conf = "  {0}%".format(round(100 - confidence))

                cv2.putText(nframe, str(id), (x+5, y-5),
                            font, 1, (255, 255, 255), 2)
                cv2.putText(nframe, str(conf), (x+5, y+h-5),
                            font, 1, (255, 255, 0), 1)
                if not str(id) in PrsList:
                    PrsList.append(str(id))

            except:
                la = 2

            prediction = np.argmax(model.predict(gray), 1)
            print(prediction)
            cv2.imshow('result', nframe)
            c = cv2.waitKey(1)
            if c & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    mark(PrsList, strtme)
