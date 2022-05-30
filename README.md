# Real Time Attendence System

#### A real time face recognition attendence marker based on CNN

## #1 Prerequisites

libs to be installed :

- opencv & opencv-contrib
- pandas
- numpy
- augmentor
- tensorflow
- pillow
- keras
- dlib
- csv

## #2 Run

run `main.py`

```sh
python main.py
```

Select what you want to do :

```sh
[1] Open Camera
[2] Capture Faces & Make Dataset
[3] Train Images
[4] Recognize & Attendance
[5] Quit
Enter Choice:
```

### #2.1 What will happen when you enter a choice

- #### Choice 1

  Open a camera instance to check your camera if it's working or what camera is selected if you have multiple camera connected

- #### Choice 2

  Start face capturing from the active camera and store them in temporal directory then apply **Data Augmentation** on captured faces then put the results on `dataset` directory and remove temporal directory

- #### Choice 3

  Train the **CNN model** on the images collected on `dataset` directory and save on `trainedModel` directory as `trained_model.h5`

- #### Choice 4
  Start **Real-Time** multi **face-recognition** and record the present persons, after stoping the system will extarct the absent persons and save same info and pack everything on `attendence` directory as `.csv` as `{date-of-today}.csv`
  <br>[see more](./attendence/important.md) about Attendence
- #### Choice 5
  Exit the program

**note** : a camera window will be shown when interacting with camera-related choices

## #3 Features on-mind

- Location-based and time-based student list selector
- GUI

## #4 Help?

contact

1. devmoussac93@gmail.com
2. yacinesha@gmail.com
