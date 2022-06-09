from PIL import Image
import os
import numpy as np
import os
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from Model import model as md
from tensorflow.keras import callbacks
from scripts.plot import ConfMatrix
from scripts.plot import plotHistory


def downsample_image(img):
    img = Image.fromarray(img.astype("uint8"), "L")
    img = img.resize((32, 32), Image.ANTIALIAS)
    return np.array(img)


# function to get the images and label data
def getImagesAndLabels(path):

    # path = "dataset"
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    ids = []

    for imagePath in imagePaths:

        # in case if error don't stop
        try:
            PIL_img = Image.open(imagePath).convert("L")  # convert it to grayscale
        except:
            continue
        img_numpy = np.array(PIL_img, "uint8")

        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faceSamples.append(img_numpy)
        ids.append(id)
    return faceSamples, ids


def train(srcpath):
    path = os.path.join(srcpath, "trainedModel")
    isdir = os.path.isdir(path)

    if not isdir:
        os.mkdir(path)

    # Path for face image database
    path = os.path.join(srcpath, "dataset")

    # recognizer = cv2.face.LBPHFaceRecognizer_create()
    # detector = cv2.CascadeClassifier(
    #     os.path.join(srcpath, "haarcascade/haarcascade_frontalface_default.xml")
    # )

    print("\nTraining Model...")
    faces, ids = getImagesAndLabels(os.path.join(srcpath, "dataset"))

    K.clear_session()
    n_faces = len(set(ids))
    model = md((32, 32, 1), n_faces)
    faces = np.asarray(faces)
    faces = np.array([downsample_image(ab) for ab in faces])
    ids = np.asarray(ids)
    faces = faces[:, :, :, np.newaxis]
    print("Shape of Data: " + str(faces.shape))
    print("Number of unique faces : " + str(n_faces))

    ids = to_categorical(ids)

    faces = faces.astype("float32")
    faces /= 255.0

    x_train, x_test, y_train, y_test = train_test_split(
        faces, ids, test_size=0.3, random_state=0
    )

    checkpoint = callbacks.ModelCheckpoint(
        "trained_model.h5",
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    )

    history = model.fit(
        x_train,
        y_train,
        batch_size=32,
        epochs=35,
        verbose=2,
        validation_data=(x_test, y_test),
        shuffle=True,
        callbacks=[checkpoint],
    )

    model.save(os.path.join(srcpath, "trainedModel/trained_model.h5"))
    # Print the numer of faces trained and end program
    print("\nTerminated with " + str(n_faces) + " faces trained")
    exportTo = os.path.join(srcpath, "exported")
    plotHistory(history, exportTo)
    # ConfMatrix(model, x_train, x_test, y_train, y_test, exportTo)
