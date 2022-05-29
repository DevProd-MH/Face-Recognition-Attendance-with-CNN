from PIL import Image
import os
import numpy as np
import cv2
import os
import h5py
import dlib
from imutils import face_utils
from tensorflow.keras.models import load_model
import sys
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split


def model(input_shape, num_classes):

    # Build the network model
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation("relu"))

    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(Conv2D(64, (1, 1)))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(Conv2D(128, (3, 3)))
    model.add(Dropout(0.5))
    model.add(Activation("relu"))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (1, 1)))
    model.add(Activation("relu"))

    model.add(Flatten())
    model.add(Dense(32))
    model.add(Dense(num_classes))
    model.add(Activation("softmax"))

    model.compile(loss='categorical_crossentropy',
                  optimizer='RMSprop',
                  metrics=['accuracy'])

    model.summary()
    return model
