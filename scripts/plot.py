import itertools
import os
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt


def plotHistory(history, exportTo):
    path = exportTo
    title = "Model accuracy"
    # # list all data in history
    # print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title(title)
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    # plt.show()
    plt.savefig(os.path.join(path, title + str(".png")))
    plt.close()

    # summarize history for loss
    title = "Model loss"
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title(title)
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    # plt.show()
    plt.savefig(os.path.join(path, title + str(".png")))
    plt.close()


def ConfMatrix(model, x_train, x_test, y_train, y_test, path):
    predicted = np.array(model.predict(x_test))
    print(predicted)
    print(y_test)
    ynew = model.predict_classes(x_test)

    Acc = accuracy_score(y_test, ynew)
    print("accuracy : ")
    print(Acc)
    tn, fp, fn, tp = confusion_matrix(np.array(y_test), ynew).ravel()
    cnf_matrix = confusion_matrix(np.array(y_test), ynew)
    y_test1 = to_categorical(y_test, 20)
    print("Confusion matrix, without normalization")
    print(cnf_matrix)
    plt.figure()
    plot_confusion_matrix(
        cnf_matrix[1:10, 1:10],
        classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        title="Confusion matrix, without normalization",
        path=path,
    )
    plt.savefig(os.path.join(path, title="Confusion matrix" + str(".png")))
    plt.close()

    plt.figure()
    plot_confusion_matrix(
        cnf_matrix[11:20, 11:20],
        classes=[10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        title="Confusion matrix, without normalization",
        path=path,
    )
    plt.savefig(os.path.join(path, title="Confusion matrix" + str(".png")))
    plt.close()

    print("Confusion matrix:\n%s" % confusion_matrix(np.array(y_test), ynew))
    print(classification_report(np.array(y_test), ynew))


def plot_confusion_matrix(
    cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues, path=""
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    print(cm)
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    # plt.show()
    plt.savefig(os.path.join(path, title + str(".png")))
