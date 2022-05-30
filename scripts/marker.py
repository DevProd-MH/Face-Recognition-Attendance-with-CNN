import csv
import os
from datetime import date
from datetime import datetime
import pandas as pd


path = 'attendence/' + str(date.today()) + '.csv'
abslist = []


def getAbsents(presnt):
    with open('attendence/studentList.csv', mode='r')as file:
        csvFile = csv.reader(file)
        for lines in sorted(csvFile):
            if not lines[0] in presnt:
                abslist.append(lines[0])


def equalList(lst, deflst):
    diff = abs(len(lst) - len(abslist))
    if (len(lst) >= len(abslist)):
        for f in range(diff):
            abslist.append(' ')
        for x in range(len(lst)-len(deflst)):
            deflst.append(' ')
    else:
        for f in range(diff):
            lst.append(' ')
        for x in range(len(abslist)-len(deflst)):
            deflst.append(' ')


def mark(prlist, startTime):
    sorted(prlist)
    getAbsents(prlist)
    info = [
        ' # Attendence started ',
        'Session : {t}'.format(t=startTime.now().strftime("%H:%M:%S")),
        'Season : {t} / {n}'.format(t=date.today().year,
                                    n=(int(date.today().year) + 1)),
        'Total Attendences : {t}'.format(t=len(prlist)),
        'Total Absents : {t}'.format(t=len(abslist))
    ]
    equalList(prlist, info)
    dataframe = pd.DataFrame({
        '#': info,
        'P': sorted(prlist),
        'A': sorted(abslist)

    })

    if not os.path.isfile(path):
        dataframe.to_csv(path, index=False)
    else:
        dataframe.to_csv(path, mode='a', index=False, header=False)

    current_time = datetime.now().strftime("%H:%M:%S")
    dataframe = pd.DataFrame({
        '#': [' ', ' # Attendence ended {t}'.format(t=current_time), ' ', ' '],
        'P': [' ', '##############', ' ', ' '],
        'A': [' ', '##############', ' ', ' ']
    })
    dataframe.to_csv(path, mode='a', index=False, header=False)
