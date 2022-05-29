import cv2
import os
import os.path
import Augmentor
import shutil
import pandas as pd
import csv


def makeDataset(srcpath):
    path = os.path.join(srcpath, 'dataset')
    if not os.path.isdir(path):
        os.mkdir(path)
        os.mkdir(os.path.join(srcpath, 'list'))

    if len(os.listdir(path)) == 0:
        print("\n\n### it's your first time using this application. Enjoy!. ###")

    os.mkdir(os.path.join(srcpath, 'tmp'))
    cam = cv2.VideoCapture(0)
    face_detector = cv2.CascadeClassifier(
        'haarcascade/haarcascade_frontalface_default.xml')

    # For each person, enter it's name and id will be auto-generated
    name = input("\nEnter Person's Name : ")
    face_id = 0
    filename = os.path.join(srcpath, "list/ids.csv")
    if not os.path.isfile(filename):
        with open(filename, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(str(face_id))
    else:
        with open(filename, mode='r') as csvfile:
            csvFile = csv.reader(csvfile)
            for lines in csvFile:
                face_id = int(lines[0]) + 1

        with open(filename, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(str(face_id))

    csvPath = os.path.join(srcpath, 'list/idname.csv')
    df = pd.DataFrame({name})

    isFile = os.path.isfile(csvPath)
    if not isFile:
        df.to_csv(r'%s', index=False, header=False)
    else:
        df.to_csv(csvPath, mode='a', index=False, header=False)

    print("\nStarting Face Capture...")
    # Initialize individual sampling face count
    count = 0

    while(True):

        ret, img = cam.read()
        img = cv2.flip(img, 1)  # flip video image vertically
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:

            cv2.rectangle(img, (x, y), (x+w+50, y+h+50), (255, 0, 0), 2)
            count += 1

            # Save the captured image into the tmp folder
            gray = gray[y:y+h, x:x+w]

            cv2.imwrite("tmp/User." + str(face_id) +
                        '.' + str(count) + ".jpg", gray)

            cv2.imshow('image', img)

        k = cv2.waitKey(100) & 0xff  # Press 'ESC' for exiting video
        if k == 27:
            break
        elif count >= 70:  # Take 70 face sample and stop video
            break

    cam.release()
    cv2.destroyAllWindows()
    print("\nFace Capture Terminated.")

    print("\nStarting data Augmentation...\n###\n")
    p = Augmentor.Pipeline('tmp')
    p.flip_left_right(0.5)
    p.flip_random(0.5)
    p.random_distortion(probability=1, grid_width=4,
                        grid_height=4, magnitude=8)
    p.skew(0.4, 0.5)
    p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
    p.zoom(probability=0.5, min_factor=1.1, max_factor=1.5)
    p.process()
    p.sample(300, multi_threaded=True)

    outpath = "tmp/output"
    for f in os.listdir(outpath):
        count += 1
        os.rename(os.path.join(outpath, f), "tmp/User." +
                  str(face_id) + '.' + str(count) + ".jpg")
    shutil.rmtree(outpath)
    outpath = "tmp"
    for f in os.listdir(outpath):
        shutil.move(os.path.join(outpath, f), "dataset")

    shutil.rmtree(outpath)
    print('\n###\nData Augmentation Terminated.')
