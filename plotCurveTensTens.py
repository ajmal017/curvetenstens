# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 18:56:40 2015

@author: markngsayyao

plot prices vs. time
"""


import csv
import datetime
import pandas as pd
import matplotlib.pyplot as plt


# store data from csv file into array
def createTimeSeries(csvFile):
    timeSeries = []
    isHeader = True
    """
    required csv format
    Date Time (yyyy-mm-dd hh:mm), Open, High, Low, Close, RSI
    """
    for csvRow in csvFile:
        if isHeader is False:
            # Date Time
            csvRow[0] = datetime.datetime(int(csvRow[0][0:4]),
                                          int(csvRow[0][5:7]),
                                          int(csvRow[0][8:10]),
                                          int(csvRow[0][11:13]),
                                          int(csvRow[0][14:16]))
            # Open
            csvRow[1] = float(csvRow[1])
            # High
            csvRow[2] = float(csvRow[2])
            # Low
            csvRow[3] = float(csvRow[3])
            # Close
            csvRow[4] = float(csvRow[4])
            # RSI
            csvRow[5] = float(csvRow[5])
            timeSeries.append(csvRow)
        else:
            isHeader = False
    return timeSeries


# open file
fileName = "CurveTensTens_Cleaned.csv"
# read file
csvFile = csv.reader(open(fileName,"r"))
# place data into list
timeSeries = createTimeSeries(csvFile)

df = pd.DataFrame(timeSeries, columns = ["dt","O","H","L","C","RSI"])

plt.grid(b = True, which = "major",color = "gray")
plt.plot(df["dt"],df["C"])
plt.savefig('ctt_plt.pdf', format='pdf')
print "success!"