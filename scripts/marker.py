import csv
import os
from datetime import date
from datetime import datetime
from numpy import full
import pandas as pd


path = "attendence/" + str(date.today()) + ".csv"
abslist = []
info = []
names_upper = []


def symetry(vrfy, orig):
    found = []
    if len(vrfy) == len(orig):
        for o in orig:
            for v in vrfy:
                if v == o:
                    found.append("true")
                else:
                    found.append("false")
        return found.count("true") == found.count("false")
    else:
        return


def getAbsents(presnt):
    with open("attendence/studentList.csv", mode="r") as file:
        csvFile = csv.reader(file)
        full_list = sorted([f[0].upper() for f in csvFile])
        for lines in full_list:
            for line in presnt:
                if symetry(line.split(" "), lines.split(" ")):
                    full_list.remove(lines)
                    break
        for n in full_list:
            abslist.append(n)


def equalList(size):
    for i in range(max(size) - len(info)):
        info.append(" ")
    for i in range(max(size) - len(abslist)):
        abslist.append(" ")
    for i in range(max(size) - len(names_upper)):
        names_upper.append(" ")


def mark(prlist, startTime, level, clsrom):
    prename = [names.upper() for names in prlist]
    for name in sorted(prename):
        names_upper.append(name.upper())
    getAbsents(names_upper)
    info_i = [
        " # Attendence started ",
        "{t}".format(t=startTime),
        "Level : {t}".format(t=level),
        "Classroom (Group) : {t}".format(t=clsrom),
        "Season : {t} / {n}".format(
            t=date.today().year, n=(int(date.today().year) + 1)
        ),
        "Total Attendences : {t}".format(t=len(names_upper)),
        "Total Absents : {t}".format(t=len(abslist)),
    ]
    for nfo in info_i:
        info.append(nfo)
    equalList([len(prlist), len(info), len(abslist)])
    dataframe = pd.DataFrame({"#": info, "P": names_upper, "A": abslist})
    if not os.path.isfile(path):
        dataframe.to_csv(path, index=False)
    else:
        dataframe.to_csv(path, mode="a", index=False, header=False)

    current_time = datetime.now().strftime("%H:%M:%S")
    dataframe = pd.DataFrame(
        {
            "#": [
                " ",
                " # Attendence ended ",
                "{t}".format(t=current_time),
                "##############",
                " ",
                " ",
            ],
            "P": [" ", "##############", "##############", "##############", " ", " "],
            "A": [" ", "##############", "##############", "##############", " ", " "],
        }
    )
    dataframe.to_csv(path, mode="a", index=False, header=False)
