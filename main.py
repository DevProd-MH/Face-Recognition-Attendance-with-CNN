import os
import shutil
import scripts.open_camera as open_camera
import scripts.dataset as dataset
import scripts.recognition as reco
import scripts.train as train


path = os.path.realpath(os.path.dirname(__file__))


def title_bar():

    print("\t**********************************************")
    print("\t*****     Real Time Attendance System    *****")
    print("\t**********************************************")


def mainMenu():

    path = os.path.realpath(os.path.dirname(__file__))
    title_bar()
    print()
    print(10 * "*", "WELCOME", 10 * "*")
    print("[1] Open Camera")
    print("[2] Capture Faces & Make Dataset")
    # print("         # add a person without training the model")
    # print("[3] Capture Faces & Make Dataset & Train Images")
    # print("         # train the model after adding a person to dataset")
    print("[3] Train Images")
    print("[4] Recognize & Attendance")
    print("[5] Reset all data")
    print("[6] Quit")
    index = 0

    while True:
        try:
            choice = int(input("Enter Choice: "))
            match choice:
                case 1:

                    openCamera(path, index)
                    mainMenu()
                    break
                case 2:

                    CaptureFaces(path, index)
                    mainMenu()
                    break
                case 3:

                    Trainimages(path)
                    mainMenu()
                    break
                case 4:

                    RecognizeFaces(path, index)
                    mainMenu()
                    break
                case 5:
                    print("Delete attendences files manually if you think you're not in need for them")
                    for dir in ["dataset","tmp","list","trainedModel"]:
                     todel = os.path.join(path,dir)
                     if os.path.isdir(todel):
                        shutil.rmtree(todel)
                    break
                case 6:

                    break
                case default:

                    print("Invalid Choice. Enter 1-5")
            mainMenu()
        except ValueError:
            print("input choice " + str(choice))
            mainMenu()
    exit


def openCamera(path, index):
    open_camera.camer(path, index)


def CaptureFaces(path, index):
    dataset.makeDataset(path, index)


def Trainimages(path):
    train.train(path)


def RecognizeFaces(path, index):
    reco.start(path, input("Enter Level : "), input("Enter Classroom : "), index)


# try:
mainMenu()
# except:
#     tmppath = os.path.join(path, "tmp")
#     if os.path.isdir(tmppath):
#         shutil.rmtree(tmppath)
