import os
import scripts.open_camera as open_camera
import scripts.dataset as dataset
import scripts.recognition as reco
import scripts.train as train


def title_bar():

    print("\t**********************************************")
    print("\t***** Real Time Attendance System *****")
    print("\t**********************************************")


def mainMenu():

    path = os.path.realpath(os.path.dirname(__file__))
    title_bar()
    print()
    print(10 * "*", "WELCOME", 10 * "*")
    print("[1] Open Camera")
    print("[2] Capture Faces & Make Dataset")
    # print("         # add a person without training the model")
    #print("[3] Capture Faces & Make Dataset & Train Images")
    # print("         # train the model after adding one person to dataset")
    print("[3] Train Images")
    print("[4] Recognize & Attendance")
    print("[5] Quit")

    while True:
        try:
            choice = int(input("Enter Choice: "))

            if choice == 1:
                openCamera(path)
                break
            elif choice == 2:
                CaptureFaces(path)
                break
            elif choice == 3:
                Trainimages(path)
                break
            elif choice == 4:
                RecognizeFaces(path)
                break
            elif choice == 5:
                break
            else:
                print("Invalid Choice. Enter 1-5")
                mainMenu()
        except ValueError:
            print("Invalid Choice. Enter 1-5\n Try Again")
    exit


def openCamera(path):
    open_camera.camer(path)
    key = input("Enter any key to return main menu")
    mainMenu()


def CaptureFaces(path):
    dataset.makeDataset(path)
    key = input("Enter any key to return main menu")
    mainMenu()


def Trainimages(path):
    train.train(path)
    key = input("Enter any key to return main menu")
    mainMenu()


def RecognizeFaces(path):
    reco.start(path)
    key = input("Enter any key to return main menu")
    mainMenu()


mainMenu()
