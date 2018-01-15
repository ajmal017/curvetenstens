# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 15:10:16 2015

@author: markngsayyao & desmondho

Backtest module for Curve Tens Tens (Price = HTS - 2*HTS + .125*TYA):
> Backtesting of Strategy, Parametric Optimisation, Walk-Forward Optimization
> Calculation of Sortino Ratio, Optimisation based on Sortino Ratio
> Calculation of Sharpe Ratio
> Calculation of Maximal Drawdown, Optimisation based on Maximal Drawdown
"""

import csv
import datetime
import time
import dateutil
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math


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


# slice timeSeries into required time period; from startDate (inclusive) to endDate (exclusive)
def sliceTimeSeries(timeSeries, startDate, endDate):
    newTimeSeries = []
    for entry in timeSeries:
        if startDate <= entry[0]  and entry[0] < endDate:
            newTimeSeries.append(entry)
    return newTimeSeries


# backtest function (minimal details, quicker)
def backtest(timeSeries, enterLong = 35, exitLong = 53,
             enterShort = 60, exitShort = 49, stopLoss = 0.04,
             record = False, slippage = 0.005, returnList = False):
    # write trade record option
    if record is True:
        r = time.time()
        backtestLog = open("backtestLog","w")
        backtestLog.write("BACKTEST TRADE LOG: " + str(timeSeries[0][0].date()) + " to " + str(timeSeries[-1][0].date()) + "\n")
    realisedBP = [[timeSeries[0][0], 0]]
    position = None # or [ dt, price (entry), +/- 1] if filled
    stoppedOut = False
    exitTrade = False
    enterLongTrade = False
    enterShortTrade = False
    for i in xrange(len(timeSeries)):
        profit = 0
        row = timeSeries[i]
        dt = row[0]
        O = row[1]
        H = row[2]
        L = row[3]
        C = row[4]
        RSI = row[5]
        # look out for jumps
        jump = False
        if i > 0:
            rowPrevious = timeSeries[i-1]
            if L >= rowPrevious[2] + 0.02 or H <= rowPrevious[3] - 0.02:
                jump = True
        # look out for futures rollover period (no entry during rolls)
        rolls = False
        if dt.month in [3,6,9,12]:
            if 8 <= dt.day and dt.day <= 14:
                rolls = True
        # refresh stopping out history if RSI crosses out
        if stoppedOut is True:
            if RSI > enterLong and RSI < enterShort:
                stoppedOut = False
        # long entry
        if (enterLongTrade is True
        and position is None and rolls is False and jump is False and stoppedOut is False):
            position = [dt, O + slippage, 1]
            enterLongTrade = False
            if record is True:
                backtestLog.writelines("\n" + str(position[0]) + " ENTER LONG @ " + str(position[1]))
        # short entry
        if (enterShortTrade is True
        and position is None and rolls is False and jump is False and stoppedOut is False):
            position = [dt, O - slippage, -1]
            enterShortTrade = False
            if record is True:
                backtestLog.writelines("\n" + str(position[0]) + " ENTER SHORT @ " + str(position[1]))
        # exit trade
        if exitTrade is True:
            prc = position[1]
            pos = position[2]
            # exit long
            if pos == 1:
                profit = (O - slippage - prc)*100 # (1 BP = 0.01)
                position = None
                exitTrade = False
                if record is True:
                    backtestLog.writelines("\n" + str(dt) + " EXIT LONG @ " + str(O - slippage) + "\n                    P/L: " + str(round(profit,1)) + "\n")
            # exit short
            elif pos == -1:
                profit = (prc - O - slippage)*100 # (1 BP = 0.01)
                position = None
                exitTrade = False
                if record is True:
                    backtestLog.writelines("\n" + str(dt) + " EXIT SHORT @ " + str(O + slippage) + "\n                    P/L: " + str(round(profit,1)) + "\n")
        # entry signal generator
        if position is None:
            if RSI < enterLong and stoppedOut is False and jump is False:
                enterLongTrade = True
                enterShortTrade = False
            elif RSI > enterShort and stoppedOut is False and jump is False:
                enterLongTrade = False
                enterShortTrade = True
            else:
                enterLongTrade = False
                enterShortTrade = False
        # exit signal generator
        elif position is not None:
            prc = position[1]
            pos = position[2] # 1 if long, -1 if short
            if pos == 1:
                if RSI >= exitLong:
                    exitTrade = True
                if (C - prc) <= - stopLoss:
                    exitTrade = True
                    if RSI > enterLong and RSI < enterShort:
                        stoppedOut = False
                    else:
                        stoppedOut = True
            elif pos == -1:
                if RSI <= exitShort:
                    exitTrade = True
                if (prc - C) <= - stopLoss:
                    exitTrade = True
                    if RSI > enterLong and RSI < enterShort:
                        stoppedOut = False
                    else:
                        stoppedOut = True
        finalisedBP = realisedBP[-1][1] + profit
        realisedBP.append([dt,finalisedBP])
    if record is True:
        backtestLog.write("\nRealised Basis Points: " + str(round(finalisedBP,1)) + "\n")
        backtestLog.write("\nTime Elapsed: " + str(time.time() - r ) + " seconds")
        backtestLog.close()
    # default output
    if returnList is False:
        return finalisedBP
    # or output entire history
    else:
        return realisedBP


# backtest function (detailed statistics report with plots, slower)
def backtestDetailedPlot(timeSeries, enterLong = 35, exitLong = 53,
                         enterShort = 60, exitShort = 49, stopLoss = 0.04,
                         slippage = 0.005, colour = "blue", returnList = False):
    # always write trade record
    r = time.time()
    backtestLog = open("backtestLogDetailed","w")
    backtestLog.write("BACKTEST TRADE LOG:\nfrom " + str(timeSeries[0][0].date()) + " to " + str(timeSeries[-1][0].date()) + "\n")
    realisedBP = [[timeSeries[0][0], 0]]
    position = None # or [ dt, price (entry), +/- 1] if filled
    stoppedOut = False
    exitTrade = False
    enterLongTrade = False
    enterShortTrade = False
    # statistics for reporting
    totalShort, totalShortWins, totalShortLosses, totalShortBE = 0, 0, 0, 0
    totalLong, totalLongWins, totalLongLosses, totalLongBE = 0, 0, 0, 0
    maxConsecutiveWins, maxConsecutiveLosses = 0, 0
    countWins, countLosses = 0, 0
    winRecord = []
    lossRecord = []
    breakEvenRecord = []
    # for calculating duration of trade
    timeRecord = []
    startTrade = None
    endTrade = None
    for i in xrange(len(timeSeries)):
        profit = 0
        row = timeSeries[i]
        dt = row[0]
        O = row[1]
        H = row[2]
        L = row[3]
        C = row[4]
        RSI = row[5]
        # look out for jumps
        jump = False
        if i > 0:
            rowPrevious = timeSeries[i-1]
            if L >= rowPrevious[2] + 0.02 or H <= rowPrevious[3] - 0.02:
                jump = True
                #backtestLog.write("\n" + str(dt) + " JUMP")
        # look out for futures rollover period (no entry during rolls)
        rolls = False
        if dt.month in [3,6,9,12]:
            if 8 <= dt.day and dt.day <= 14:
                rolls = True
        # refresh stopping out history if RSI crosses out
        if stoppedOut is True:
            if RSI > enterLong and RSI < enterShort:
                stoppedOut = False
        # long entry
        if (enterLongTrade is True
        and position is None and rolls is False and jump is False and stoppedOut is False):
            startTrade = dt
            position = [dt, O + slippage, 1]
            enterLongTrade = False
            backtestLog.writelines("\n" + str(position[0]) + " ENTER LONG @ " + str(position[1]))
            totalLong += 0.5
        # short entry
        if (enterShortTrade is True
        and position is None and rolls is False and jump is False and stoppedOut is False):
            startTrade = dt
            position = [dt, O - slippage, -1]
            enterShortTrade = False
            backtestLog.writelines("\n" + str(position[0]) + " ENTER SHORT @ " + str(position[1]))
            totalShort += 0.5
        # exit trade
        if exitTrade is True:
            prc = position[1]
            pos = position[2]
            endTrade = dt
            timeRecord.append((endTrade - startTrade).total_seconds()/3600)
            startTrade = None
            endTrade = None
            # exit long
            if pos == 1:
                profit = (O - slippage - prc)*100 # (1 BP = 0.01)
                position = None
                exitTrade = False
                backtestLog.writelines("\n" + str(dt) + " EXIT LONG @ " + str(O - slippage) + "\n                    P/L: " + str(round(profit,1)) + "\n")
                totalLong += 0.5
                if profit > 0:
                    winRecord.append(profit)
                    countWins += 1
                    maxConsecutiveLosses = max(maxConsecutiveLosses,countLosses)
                    countLosses = 0
                    totalLongWins += 1
                elif profit < 0:
                    lossRecord.append(profit)
                    maxConsecutiveWins = max(maxConsecutiveWins,countWins)
                    countWins = 0
                    countLosses += 1
                    totalLongLosses += 1
                else:
                    breakEvenRecord.append(profit)
                    totalLongBE += 1
            # exit short
            elif pos == -1:
                profit = (prc - O - slippage)*100 # (1 BP = 0.01)
                position = None
                exitTrade = False
                backtestLog.writelines("\n" + str(dt) + " EXIT SHORT @ " + str(O + slippage) + "\n                    P/L: " + str(round(profit,1)) + "\n")
                totalShort += 0.5
                if profit > 0:
                    winRecord.append(profit)
                    countWins += 1
                    maxConsecutiveLosses = max(maxConsecutiveLosses,countLosses)
                    countLosses = 0
                    totalShortWins += 1
                elif profit < 0:
                    lossRecord.append(profit)
                    maxConsecutiveWins = max(maxConsecutiveWins,countWins)
                    countWins = 0
                    countLosses += 1
                    totalShortLosses += 1
                else:
                    breakEvenRecord.append(profit)
                    totalShortBE += 1
        # entry signal generator
        if position is None:
            if RSI < enterLong and stoppedOut is False and jump is False:
                if rolls is False or (dt + datetime.timedelta(hours = 1)).day > 14:
                    backtestLog.writelines("\n" + str(dt) + " SIGNAL: RSI = " + str(RSI) )
                enterLongTrade = True
                enterShortTrade = False
            elif RSI > enterShort and stoppedOut is False and jump is False:
                if rolls is False or (dt + datetime.timedelta(hours = 1)).day > 14:
                    backtestLog.writelines("\n" + str(dt) + " SIGNAL: RSI = " + str(RSI) )
                enterLongTrade = False
                enterShortTrade = True
            else:
                enterLongTrade = False
                enterShortTrade = False
        # exit signal generator
        elif position is not None:
            prc = position[1]
            pos = position[2] # 1 if long, -1 if short
            if pos == 1:
                if RSI >= exitLong:
                    exitTrade = True
                    backtestLog.writelines("\n" + str(dt) + " SIGNAL: RSI = " + str(RSI) )
                if (C - prc) <= - stopLoss:
                    exitTrade = True
                    backtestLog.writelines("\n" + str(dt) + " SIGNAL: STOP")
                    if RSI > enterLong and RSI < enterShort:
                        stoppedOut = False
                    else:
                        stoppedOut = True
            elif pos == -1:
                if RSI <= exitShort:
                    exitTrade = True
                    backtestLog.writelines("\n" + str(dt) + " SIGNAL: RSI = " + str(RSI) )
                if (prc - C) <= - stopLoss:
                    exitTrade = True
                    backtestLog.writelines("\n" + str(dt) + " SIGNAL: STOP")
                    if RSI > enterLong and RSI < enterShort:
                        stoppedOut = False
                    else:
                        stoppedOut = True
        finalisedBP = realisedBP[-1][1] + profit
        realisedBP.append([dt,finalisedBP])
    # calculate maximal drawdown & duration
    maximalDrawdown = 0
    maximalDrawdownDuration = 0
    firstDuration = None
    secondDuration = None
    first = 0
    second = 0
    for i in xrange(len(realisedBP)):
        # base
        if i == 0:
            first = realisedBP[0][1]
            second = realisedBP[0][1]
            firstDuration = realisedBP[0][0]
            secondDuration = realisedBP[0][0]
        else:
            # drop
            if second >= realisedBP[i][1]:
                second = realisedBP[i][1]
                secondDuration = realisedBP[i][0]
            else:
                drawdown = first - second
                duration = (secondDuration - firstDuration).total_seconds()/3600
                if maximalDrawdown > drawdown:
                    first = realisedBP[i][1]
                    second = realisedBP[i][1]
                    firstDuration = realisedBP[i][0]
                    secondDuration = realisedBP[i][0]
                elif maximalDrawdown == drawdown:
                    maximalDrawdown = drawdown
                    maximalDrawdownDuration = max(maximalDrawdownDuration,duration)
                    # refresh comparision
                    first = realisedBP[i][1]
                    second = realisedBP[i][1]
                    firstDuration = realisedBP[i][0]
                    secondDuration = realisedBP[i][0]
                else:
                    maximalDrawdownDuration = duration
                    maximalDrawdown = drawdown
                    # refresh comparision
                    first = realisedBP[i][1]
                    second = realisedBP[i][1]
                    firstDuration = realisedBP[i][0]
                    secondDuration = realisedBP[i][0]
            if i == len(realisedBP)-1:
                if maximalDrawdown > drawdown:
                    continue
                elif maximalDrawdown == drawdown:
                    maximalDrawdown = drawdown
                    maximalDrawdownDuration = max(maximalDrawdownDuration,duration)
                else:
                    maximalDrawdown = drawdown
                    maximalDrawdownDuration = duration
    # calculate maximal drawup & duration
    maximalDrawup = 0
    maximalDrawupDuration = 0
    firstDuration = None
    secondDuration = None
    first = 0
    second = 0
    for i in xrange(len(realisedBP)):
        # base
        if i == 0:
            first = realisedBP[0][1]
            second = realisedBP[0][1]
            firstDuration = realisedBP[0][0]
            secondDuration = realisedBP[0][0]
        else:
            # up
            if second <= realisedBP[i][1]:
                second = realisedBP[i][1]
                secondDuration = realisedBP[i][0]
            else:
                drawup = second - first
                duration = (secondDuration - firstDuration).total_seconds()/3600
                if maximalDrawup > drawup:
                    first = realisedBP[i][1]
                    second = realisedBP[i][1]
                    firstDuration = realisedBP[i][0]
                    secondDuration = realisedBP[i][0]
                elif maximalDrawup == drawup:
                    maximalDrawup = drawup
                    maximalDrawupDuration = max(maximalDrawupDuration, duration)
                    # refresh comparision
                    first = realisedBP[i][1]
                    second = realisedBP[i][1]
                    firstDuration = realisedBP[i][0]
                    secondDuration = realisedBP[i][0]
                else:
                    maximalDrawupDuration = duration
                    maximalDrawup = drawup
                    # refresh comparision
                    first = realisedBP[i][1]
                    second = realisedBP[i][1]
                    firstDuration = realisedBP[i][0]
                    secondDuration = realisedBP[i][0]
            if i == len(realisedBP)-1:
                maximalDrawup = max(maximalDrawup,second - first, realisedBP[i][1] - first)
                if maximalDrawup > drawup:
                    continue
                elif maximalDrawdown == drawup:
                    maximalDrawup = drawup
                    maximalDrawupDuration = max(maximalDrawupDuration,duration)
                else:
                    maximalDrawup = drawup
                    maximalDrawupDuration = duration
    # calculate average win,loss,profit per trade
    averageWin = round(np.mean(winRecord),2)
    averageLoss = round(np.mean(lossRecord),2)
    fullRecord = np.array(winRecord + lossRecord + breakEvenRecord)
    averageTrade = round(np.mean(fullRecord),2)
    varianceTrade = round(np.var(fullRecord),2)
    medianTrade = round(np.median(fullRecord),2)
    pWin = round(float(len(winRecord)*100)/len(fullRecord),2)
    pLoss = round(float(len(lossRecord)*100)/len(fullRecord),2)
    pBreakEven = round(float(len(breakEvenRecord)*100)/len(fullRecord),2)
    largestWin = np.max(fullRecord)
    largestLoss = np.min(fullRecord)
    backtestLog.write("\nParameters Used: "  + str([enterLong, exitLong, enterShort,exitShort,stopLoss*100]))
    backtestLog.write("\nAssumed slippage of " + str(slippage*100) + " Basis Points per side." )
    backtestLog.write("\n\nTrade Profit/Loss Statistics (in Basis Points):")
    backtestLog.write("\nTotal Realised Basis Points, " + str(round(finalisedBP,1)))
    backtestLog.write("\nTrade P/L Mean, " + str(averageTrade) +
                      "\nTrade P/L Variance, " + str(varianceTrade) +
                      "\nTrade P/L Median, " + str(medianTrade))
    backtestLog.write("\n\nWins & Losses (in Basis Points):")
    backtestLog.write("\nWin/Loss Mean, " + str(averageWin) + "/" + str(averageLoss))
    backtestLog.write("\nWin/Loss Variance, " + str(round(np.var(winRecord),2)) + "/" + str(round(np.var(lossRecord),2)))
    backtestLog.write("\nWin/Loss Median, " + str(np.median(winRecord)) + "/" + str(np.median(lossRecord)))
    backtestLog.write("\nLargest Win/Loss, " + str(largestWin) + "/" + str(largestLoss))
    backtestLog.write("\n\nTrade Statistics:")
    backtestLog.write("\nTotal Trades, " + str(totalLong + totalShort))
    backtestLog.write("\nWin/Loss/Break-Even, " +
                      str(totalLongWins+totalShortWins) + "/" +
                      str(totalLongLosses+totalShortLosses) + "/" +
                      str(totalLongBE+totalShortBE))
    backtestLog.write("\nPercentage Win/Loss/Break-Even, " +
                      str(pWin) + "% / " +
                      str(pLoss) + "% / " +
                      str(pBreakEven) + "%")
    backtestLog.write("\nAverage Duration of Trade, " + str(round(np.mean(timeRecord),2)) + " hours")
    pLongWin = round(totalLongWins/totalLong*100,2)
    pLongLoss = round(totalLongLosses/totalLong*100,2)
    pLongBreakEven = round(totalLongBE/totalLong*100,2)
    backtestLog.write("\n\nLong Trade Statistics:\nTotal Long Trades, " + str(totalLong) +
                      "\nWin/Loss/Break-Even, " + str(totalLongWins) + "/" + str(totalLongLosses) + "/" + str(totalLongBE) +
                      "\nPercentage Win/Loss/Break-Even, " +
                      str(pLongWin) + "% / " +
                      str(pLongLoss) + "% / " +
                      str(pLongBreakEven) + "%")
    pShortWin = round(totalShortWins/totalShort*100,2)
    pShortLoss = round(totalShortLosses/totalShort*100,2)
    pShortBreakEven = round(totalShortBE/totalShort*100,2)
    backtestLog.write("\n\nShort Trade Statistics:\nTotal Short Trades, " + str(totalShort) +
                      "\nWin/Loss/Break-Even, " + str(totalShortWins) + "/" + str(totalShortLosses) + "/" + str(totalShortBE) +
                      "\nPercentage Win/Loss/Break-Even, " +
                      str(pShortWin) + "% / " +
                      str(pShortLoss) + "% / " +
                      str(pShortBreakEven) + "%")
    backtestLog.write("\n\nPeak-Trough Statistics (in Basis Points):\nMaximal Drawdown (Duration), -" +
                      str(maximalDrawdown) + " (" + str(round(maximalDrawdownDuration,2)) + " hours)" +
                      "\nMaximal Draw-up (Duration), " +
                      str(maximalDrawup) + " (" + str(round(maximalDrawupDuration,2)) + " hours)" )
    backtestLog.write("\nMaximum Consecutive Wins/Losses, " + str(maxConsecutiveWins) + "/" + str(maxConsecutiveLosses))

    backtestLog.write("\n\nTime Elapsed: " + str(time.time() - r) + " seconds")
    backtestLog.close()
    df = pd.DataFrame(realisedBP,columns = ["dt","bp"])
    plt.plot(df["dt"],df["bp"],colour)
    plt.grid(b = True, which = "major", color = "gray")
    plt.title("Performance (in Basis Points) versus Time \n")
    if returnList:
        return realisedBP
    else:
        return realisedBP[-1]


# plot relationship between slippage and performance
def plotSlippageChart():
    # load file and create time series
    fileName = "CurveTT_Ripped_Long.csv"
    csvFile = csv.reader(open(fileName,"r"))
    timeSeries = createTimeSeries(csvFile)
    # generate results
    slippage00 = backtest(timeSeries, enterLong = 31, exitLong = 53, enterShort = 60, exitShort = 41, stopLoss = 0.05, slippage = 0, returnList = True)
    slippage01 = backtest(timeSeries, enterLong = 31, exitLong = 53, enterShort = 60, exitShort = 41, stopLoss = 0.05, slippage = 0.001, returnList = True)
    slippage02 = backtest(timeSeries, enterLong = 31, exitLong = 53, enterShort = 60, exitShort = 41, stopLoss = 0.05, slippage = 0.002, returnList = True)
    slippage03 = backtest(timeSeries, enterLong = 31, exitLong = 53, enterShort = 60, exitShort = 41, stopLoss = 0.05, slippage = 0.003, returnList = True)
    slippage04 = backtest(timeSeries, enterLong = 31, exitLong = 53, enterShort = 60, exitShort = 41, stopLoss = 0.05, slippage = 0.004, returnList = True)
    slippage05 = backtest(timeSeries, enterLong = 31, exitLong = 53, enterShort = 60, exitShort = 41, stopLoss = 0.05, slippage = 0.005, returnList = True)
    slippage06 = backtest(timeSeries, enterLong = 31, exitLong = 53, enterShort = 60, exitShort = 41, stopLoss = 0.05, slippage = 0.006, returnList = True)
    slippage07 = backtest(timeSeries, enterLong = 31, exitLong = 53, enterShort = 60, exitShort = 41, stopLoss = 0.05, slippage = 0.007, returnList = True)
    slippage08 = backtest(timeSeries, enterLong = 31, exitLong = 53, enterShort = 60, exitShort = 41, stopLoss = 0.05, slippage = 0.008, returnList = True)
    slippage09 = backtest(timeSeries, enterLong = 31, exitLong = 53, enterShort = 60, exitShort = 41, stopLoss = 0.05, slippage = 0.009, returnList = True)
    slippage10 = backtest(timeSeries, enterLong = 31, exitLong = 53, enterShort = 60, exitShort = 41, stopLoss = 0.05, slippage = 0.010, returnList = True)
    # store in pandas datafram
    df00 = pd.DataFrame(slippage00, columns = ["dt","bp"])
    df01 = pd.DataFrame(slippage01, columns = ["dt","bp"])
    df02 = pd.DataFrame(slippage02, columns = ["dt","bp"])
    df03 = pd.DataFrame(slippage03, columns = ["dt","bp"])
    df04 = pd.DataFrame(slippage04, columns = ["dt","bp"])
    df05 = pd.DataFrame(slippage05, columns = ["dt","bp"])
    df06 = pd.DataFrame(slippage06, columns = ["dt","bp"])
    df07 = pd.DataFrame(slippage07, columns = ["dt","bp"])
    df08 = pd.DataFrame(slippage08, columns = ["dt","bp"])
    df09 = pd.DataFrame(slippage09, columns = ["dt","bp"])
    df10 = pd.DataFrame(slippage10, columns = ["dt","bp"])
    # plot on the same chart
    plt.title("Performance in Basis Points\n")
    plt.grid(b = True, which = "major", color = "gray")
    plt.plot(df00["dt"], df00["bp"], "black")
    plt.plot(df01["dt"], df01["bp"], "darkblue")
    plt.plot(df02["dt"], df02["bp"], "darkcyan")
    plt.plot(df03["dt"], df03["bp"], "darkgoldenrod")
    plt.plot(df04["dt"], df04["bp"], "darkgray")
    plt.plot(df05["dt"], df05["bp"], "darkgreen")
    plt.plot(df06["dt"], df06["bp"], "darkkhaki")
    plt.plot(df07["dt"], df07["bp"], "darkmagenta")
    plt.plot(df08["dt"], df08["bp"], "darkolivegreen")
    plt.plot(df09["dt"], df09["bp"], "darkorange")
    plt.plot(df10["dt"], df10["bp"], "darkorchid")
    plt.legend([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], numpoints = 1, loc='center left', bbox_to_anchor=(1.0, 0.75), title = "Slippage per Side")
    return None


# calulate Sortino Ratio from backtest results
def calculateSortinoRatio(results,interval = 30, benchmark = 0.25):
    periodicReturns = []
    periodicDownsideRisk = []
    startDate = results[0][0]
    startBP = results[0][1]
    endBP = startBP
    endDate  = startDate + datetime.timedelta(days = interval)
    index = 0
    lastDate = results[-1][0]
    while index < len(results):
        while results[index][0].date() < endDate.date():
            index += 1
            endBP = results[index][1]
        #returns = (endBP - startBP)/0.54 - benchmark
        """
        /0.54 is approximately *1.851851
        """
        returns = (endBP - startBP)*1.851851 - benchmark
        # note excess return
        periodicReturns.append(returns)
        # note negative excess returns
        if returns < 0:
            periodicDownsideRisk.append(returns*returns)
        else:
            periodicDownsideRisk.append(0)
        # prepare next interval
        startDate = endDate
        endDate = startDate + datetime.timedelta(days = interval)
        startBP = endBP
        if endDate > lastDate:
            break
        else:
            continue
    sd = math.sqrt(np.sum(periodicDownsideRisk)/(len(periodicDownsideRisk)-1))
    if sd == 0:
        return np.mean(periodicReturns)*100000
    else:
        return np.mean(periodicReturns)/sd


# calulate Sharpe Ratio from backtest results
def calculateSharpeRatio(results,interval = 30, benchmark = 0.25):
    periodicReturns = []
    periodicRisk = []
    startDate = results[0][0]
    startBP = results[0][1]
    endBP = startBP
    endDate  = startDate + datetime.timedelta(days = interval)
    index = 0
    lastDate = results[-1][0]
    while index < len(results):
        while results[index][0].date() < endDate.date():
            index += 1
            endBP = results[index][1]
        #returns = (endBP - startBP)/0.54 - benchmark
        """
        /0.54 is approximately *1.851851
        """
        returns = (endBP - startBP)*1.851851 - benchmark
        # note excess return
        periodicReturns.append(returns)
        # note squared excess returns
        periodicRisk.append(returns*returns)
        # prepare next interval
        startDate = endDate
        endDate = startDate + datetime.timedelta(days = interval)
        startBP = endBP
        if endDate > lastDate:
            break
        else:
            continue
    return np.mean(periodicReturns)/math.sqrt(np.sum(periodicRisk)/(len(periodicRisk)-1))


# calculate maximal drawdown form backtest results
def calculateMaximalDrawdown(results):
    maximalDrawdown = 0
    first = 0
    second  = 0
    for i in xrange(len(results)-1):
        if i == 0:
            first = results[i][1]
            second = results[i+1][1]
        else:
            if second >= results[i][1]:
                second = results[i][1]
            else:
                drawdown = first - second
                maximalDrawdown = max(maximalDrawdown,drawdown)
                first = results[i][1]
                second = results[i][1]
            if i == len(results)-1:
                maximalDrawdown = max(maximalDrawdown,first - second)
    return maximalDrawdown


# optimisation based on realised basis points
def optimise(timeSeries,
             enterLongLower = 25, enterLongUpper = 35,
             exitLongLower = 44, exitLongUpper = 59,
             enterShortLower = 60, enterShortUpper = 70,
             exitShortLower = 41, exitShortUpper = 56,
             stopLossLower = 0.04, stopLossUpper = 0.05,
             display = True):
    # just to count total cases
    totalCases = 0
    stopLoss = stopLossLower
    while stopLoss <= stopLossUpper:
        enterLong = enterLongLower
        while enterLong <= enterLongUpper:
            exitLong = exitLongLower
            while exitLong <= exitLongUpper:
                enterShort = enterShortLower
                while enterShort <= enterShortUpper:
                    exitShort = exitShortLower
                    while exitShort <= exitShortUpper:
                        totalCases += 1
                        exitShort += 1
                    enterShort += 1
                exitLong += 1
            enterLong += 1
        stopLoss += 0.01
    # start optimising
    parameters = []
    bestRealisedBasisPoints = 0
    case = 0
    t = time.time()
    stopLoss = stopLossLower
    while stopLoss <= stopLossUpper:
        enterLong = enterLongLower
        while enterLong <= enterLongUpper:
            exitLong = exitLongLower
            while exitLong <= exitLongUpper:
                enterShort = enterShortLower
                while enterShort <= enterShortUpper:
                    exitShort = exitShortLower
                    while exitShort <= exitShortUpper:
                        result = backtest(timeSeries, enterLong, exitLong,
                                          enterShort, exitShort, stopLoss)
                        if result > bestRealisedBasisPoints:
                            parameters = [[enterLong, exitLong,
                                           enterShort, exitShort, stopLoss]]
                            bestRealisedBasisPoints = result
                        del result
                        case += 1
                        if display is True:
                            if case % 1000 == 0:
                                print("Completed " + str(case) + " of " + str(totalCases) + ". Time Elapsed: " + str(time.time() - t) + "s")
                        exitShort += 1
                    enterShort += 1
                exitLong += 1
            enterLong += 1
        stopLoss += 0.01
    if display is True:
        print("Completed " + str(case) + " of " + str(totalCases) + ". Time Elapsed: " + str(time.time() - t) + "s")
    parameters.append(bestRealisedBasisPoints)
    return parameters


# optimisation based on Sortino Ratio
def optimiseSortinoRatio(timeSeries,
                         enterLongLower = 25, enterLongUpper = 35,
                         exitLongLower = 44, exitLongUpper = 59,
                         enterShortLower = 60, enterShortUpper = 70,
                         exitShortLower = 41, exitShortUpper = 56,
                         stopLossLower = 0.04, stopLossUpper = 0.05,
                         display = True):
    # just to count total cases
    totalCases = 0
    stopLoss = stopLossLower
    while stopLoss <= stopLossUpper:
        enterLong = enterLongLower
        while enterLong <= enterLongUpper:
            exitLong = exitLongLower
            while exitLong <= exitLongUpper:
                enterShort = enterShortLower
                while enterShort <= enterShortUpper:
                    exitShort = exitShortLower
                    while exitShort <= exitShortUpper:
                        totalCases += 1
                        exitShort += 1
                    enterShort += 1
                exitLong += 1
            enterLong += 1
        stopLoss += 0.01
    # start optimising
    parameters = []
    bestSR = 0
    case = 0
    t = time.time()
    stopLoss = stopLossLower
    while stopLoss <= stopLossUpper:
        enterLong = enterLongLower
        while enterLong <= enterLongUpper:
            exitLong = exitLongLower
            while exitLong <= exitLongUpper:
                enterShort = enterShortLower
                while enterShort <= enterShortUpper:
                    exitShort = exitShortLower
                    while exitShort <= exitShortUpper:
                        result = calculateSortinoRatio(backtest(timeSeries, enterLong, exitLong, enterShort, exitShort, stopLoss, returnList = True), interval = 30, benchmark = 0.25)
                        # if downside risk is zero
                        if result == np.Inf:
                            return [[enterLong, exitLong, enterShort, exitShort, stopLoss]]
                        if result > bestSR:
                            parameters = [[enterLong, exitLong, enterShort, exitShort, stopLoss],np.Inf]
                            bestSR = result
                        del result
                        case += 1
                        if display is True:
                            if case % 1000 == 0:
                                print("Completed " + str(case) + " of " + str(totalCases) + ". Time Elapsed: " + str(time.time() - t) + "s")
                        exitShort += 1
                    enterShort += 1
                exitLong += 1
            enterLong += 1
        stopLoss += 0.01
    if display is True:
        print("Completed " + str(case) + " of " + str(totalCases) + ". Time Elapsed: " + str(time.time() - t) + "s")
    parameters.append(bestSR)
    return parameters


# optimisation based on Maximal Drawdown
def optimiseMaximalDrawdown(timeSeries,
                            enterLongLower = 25, enterLongUpper = 35,
                            exitLongLower = 44, exitLongUpper = 59,
                            enterShortLower = 60, enterShortUpper = 70,
                            exitShortLower = 41, exitShortUpper = 56,
                            stopLossLower = 0.04, stopLossUpper = 0.05,
                            display = True):
    # just to count total cases
    totalCases = 0
    stopLoss = stopLossLower
    while stopLoss <= stopLossUpper:
        enterLong = enterLongLower
        while enterLong <= enterLongUpper:
            exitLong = exitLongLower
            while exitLong <= exitLongUpper:
                enterShort = enterShortLower
                while enterShort <= enterShortUpper:
                    exitShort = exitShortLower
                    while exitShort <= exitShortUpper:
                        totalCases += 1
                        exitShort += 1
                    enterShort += 1
                exitLong += 1
            enterLong += 1
        stopLoss += 0.01
    # start optimising
    parameters = []
    bestMD = None
    case = 0
    t = time.time()
    stopLoss = stopLossLower
    while stopLoss <= stopLossUpper:
        enterLong = enterLongLower
        while enterLong <= enterLongUpper:
            exitLong = exitLongLower
            while exitLong <= exitLongUpper:
                enterShort = enterShortLower
                while enterShort <= enterShortUpper:
                    exitShort = exitShortLower
                    while exitShort <= exitShortUpper:
                        result = calculateMaximalDrawdown(backtest(timeSeries, enterLong, exitLong, enterShort, exitShort, stopLoss, returnList = True))
                        if result < bestMD or bestMD is None:
                            parameters = [[enterLong, exitLong, enterShort, exitShort, stopLoss]]
                            bestMD = result
                        del result
                        case += 1
                        if display is True:
                            if case % 1000 == 0:
                                print("Completed " + str(case) + " of " + str(totalCases) + ". Time Elapsed: " + str(time.time() - t) + "s")
                        exitShort += 1
                    enterShort += 1
                exitLong += 1
            enterLong += 1
        stopLoss += 0.01
    if display is True:
        print("Completed " + str(case) + " of " + str(totalCases) + ". Time Elapsed: " + str(time.time() - t) + "s")
    parameters.append(bestMD)
    return parameters


# walk forward optimisation (based on Basis Points) back testing function
def walkforward(timeSeries, optimisationPeriod = 1, testPeriod = 1,
                enterLongLower = 25, enterLongUpper = 35,
                exitLongLower = 44, exitLongUpper = 59,
                enterShortLower = 60, enterShortUpper = 70,
                exitShortLower = 41, exitShortUpper = 56,
                stopLossLower = 0.04, stopLossUpper = 0.05,
                target = "bp"):
    """
    bp: Basis Point optimisation
    sortino: Sortino Ratio optimisation
    md: Maximal Drawdown optmimsation
    """
    outputDetails = [[optimisationPeriod,testPeriod]]
    # record actions in log
    walkforwardLog = open("walkforwardLog","w")
    walkforwardLog.write("WALK FORWARD TEST RECORD: \n\n")
    complete = False
    lastDate = timeSeries[-1][0]
    # first reference frame
    startOptimisationDate = timeSeries[0][0]
    endOptimisationDate = startOptimisationDate + dateutil.relativedelta.relativedelta(months = optimisationPeriod)
    endTestDate = endOptimisationDate + dateutil.relativedelta.relativedelta(months = testPeriod)
    if endTestDate >= lastDate:
        complete = True
    # record of basis points
    totalBP = 0
    while not complete:
        # save and continue (in case the computer crashes...)
        walkforwardLog.close()
        walkforwardLog = open("walkforwardLog","a")
        # optimise
        print("Optimising from " + str(startOptimisationDate.date()) + " to " + str(endOptimisationDate.date()) + "...")
        walkforwardLog.write("Optimising from " + str(startOptimisationDate.date()) + " to " + str(endOptimisationDate.date()) + "...\n")
        optimisationTimeSeries = sliceTimeSeries(timeSeries,startOptimisationDate,endOptimisationDate)
        # optimisation target
        if target == "bp":
            parameters = optimise(optimisationTimeSeries,
                                  enterLongLower,enterLongUpper,
                                  exitLongLower,exitLongUpper,
                                  enterShortLower,enterShortUpper,
                                  exitShortLower,exitShortUpper,
                                  stopLossLower,stopLossUpper)
        elif target == "sortino":
            parameters = optimiseSortinoRatio(optimisationTimeSeries,
                                              enterLongLower,enterLongUpper,
                                              exitLongLower,exitLongUpper,
                                              enterShortLower,enterShortUpper,
                                              exitShortLower,exitShortUpper,
                                              stopLossLower,stopLossUpper)
        elif target == "md":
            parameters = optimiseMaximalDrawdown(optimisationTimeSeries,
                                                 enterLongLower,enterLongUpper,
                                                 exitLongLower,exitLongUpper,
                                                 enterShortLower,enterShortUpper,
                                                 exitShortLower,exitShortUpper,
                                                 stopLossLower,stopLossUpper)
        else:
            print("Invalid Optimisation Target Setting: " + target)
            return None
        print("Finished Optimisation: " + str(parameters[0]) + " " + str(parameters[-1]))
        walkforwardLog.write("Finished Optimisation: " + str(parameters[0]) + " " + str(parameters[-1]) + "\n")
        # apply forward
        print("Applying parameters on the period " + str(endOptimisationDate.date()) + " to " + str(endTestDate.date()))
        walkforwardLog.write("Applying parameters on the period " + str(endOptimisationDate.date()) + " to " + str(endTestDate.date()) + "\n")
        testTimeSeries = sliceTimeSeries(timeSeries,endOptimisationDate,endTestDate)
        print("Change in Basis Points: " + str(backtest(testTimeSeries,parameters[0][0],parameters[0][1],parameters[0][2],parameters[0][3],parameters[0][4])))
        walkforwardLog.write("Change in Basis Points: " + str(backtest(testTimeSeries,parameters[0][0],parameters[0][1],parameters[0][2],parameters[0][3],parameters[0][4])) + "\n")
        totalBP += backtest(testTimeSeries,parameters[0][0],parameters[0][1],parameters[0][2],parameters[0][3],parameters[0][4])
        print("Cumulative Basis Points: " + str(totalBP) + "\n")
        walkforwardLog.write("Cumulative Basis Points: " + str(totalBP) + "\n\n")
        outputDetails.append(parameters[0])
        del optimisationTimeSeries
        del parameters
        del testTimeSeries
        # re-calibrate lookback window
        startOptimisationDate += dateutil.relativedelta.relativedelta(months = testPeriod)
        endOptimisationDate += dateutil.relativedelta.relativedelta(months = testPeriod)
        endTestDate += dateutil.relativedelta.relativedelta(months = + testPeriod)
        if endTestDate > lastDate:
            complete = True
    # save file
    walkforwardLog.close()
    return outputDetails


# backtest function for simulating walk forward strategy (detailed statistics report with plots, slower)
def backtestDetailedPlotWF(timeSeries, outputDetails,
                           slippage = 0.005, colour = "blue", returnList = False):
    # slice timeSeries appropriately; must start after first optimisation period
    s = timeSeries[0][0] + dateutil.relativedelta.relativedelta(months = outputDetails[0][0])
    e = s + dateutil.relativedelta.relativedelta(months = (len(outputDetails)-1)*outputDetails[0][1])
    timeSeries = sliceTimeSeries(timeSeries,s,e)
    # starting parameters
    parameterSet = 1
    startingFrom = timeSeries[0][0]
    enterLong = outputDetails[parameterSet][0]
    exitLong = outputDetails[parameterSet][1]
    enterShort = outputDetails[parameterSet][2]
    exitShort = outputDetails[parameterSet][3]
    stopLoss = outputDetails[parameterSet][4]
    # always write trade record
    r = time.time()
    backtestLog = open("backtestLogDetailedWF","w")
    backtestLog.write("BACKTEST TRADE LOG:\nfrom " + str(timeSeries[0][0].date()) + " to " + str(timeSeries[-1][0].date()) + "\n")
    realisedBP = [[timeSeries[0][0], 0]]
    position = None # or [ dt, price (entry), +/- 1] if filled
    stoppedOut = False
    exitTrade = False
    enterLongTrade = False
    enterShortTrade = False
    # statistics for reporting
    totalShort, totalShortWins, totalShortLosses, totalShortBE = 0, 0, 0, 0
    totalLong, totalLongWins, totalLongLosses, totalLongBE = 0, 0, 0, 0
    maxConsecutiveWins, maxConsecutiveLosses = 0, 0
    countWins, countLosses = 0, 0
    winRecord = []
    lossRecord = []
    breakEvenRecord = []
    # for calculating duration of trade
    timeRecord = []
    startTrade = None
    endTrade = None
    backtestLog.write("\nUsing Parameters " + str([enterLong,exitLong,enterShort,exitShort,stopLoss*100]))
    for i in xrange(len(timeSeries)):
        profit = 0
        row = timeSeries[i]
        dt = row[0]
        O = row[1]
        H = row[2]
        L = row[3]
        C = row[4]
        RSI = row[5]
        # update parameters, if testPeriod is over and not holding any position
        if dt >= (startingFrom + dateutil.relativedelta.relativedelta(months = outputDetails[0][1])) and position is None:
            if parameterSet < len(outputDetails) - 1:
                parameterSet += 1
                enterLong = outputDetails[parameterSet][0]
                exitLong = outputDetails[parameterSet][1]
                enterShort = outputDetails[parameterSet][2]
                exitShort = outputDetails[parameterSet][3]
                stopLoss = outputDetails[parameterSet][4]
                backtestLog.write("\nCumulative Basis Points: " + str(realisedBP[-1][1]))
                backtestLog.write("\nUsing Parameters " + str([enterLong,exitLong,enterShort,exitShort,stopLoss*100]))
            print dt
            print parameterSet
            startingFrom = startingFrom + dateutil.relativedelta.relativedelta(months = outputDetails[0][1])
        # look out for jumps
        jump = False
        if i > 0:
            rowPrevious = timeSeries[i-1]
            if L >= rowPrevious[2] + 0.02 or H <= rowPrevious[3] - 0.02:
                jump = True
                #backtestLog.write("\n" + str(dt) + " JUMP")
        # look out for futures rollover period (no entry during rolls)
        rolls = False
        if dt.month in [3,6,9,12]:
            if 8 <= dt.day and dt.day <= 14:
                rolls = True
        # refresh stopping out history if RSI crosses out
        if stoppedOut is True:
            if RSI > enterLong and RSI < enterShort:
                stoppedOut = False
        # long entry
        if (enterLongTrade is True
        and position is None and rolls is False and jump is False and stoppedOut is False):
            startTrade = dt
            position = [dt, O + slippage, 1]
            enterLongTrade = False
            backtestLog.writelines("\n" + str(position[0]) + " ENTER LONG @ " + str(position[1]))
            totalLong += 0.5
        # short entry
        if (enterShortTrade is True
        and position is None and rolls is False and jump is False and stoppedOut is False):
            startTrade = dt
            position = [dt, O - slippage, -1]
            enterShortTrade = False
            backtestLog.writelines("\n" + str(position[0]) + " ENTER SHORT @ " + str(position[1]))
            totalShort += 0.5
        # exit trade
        if exitTrade is True:
            prc = position[1]
            pos = position[2]
            endTrade = dt
            timeRecord.append((endTrade - startTrade).total_seconds()/3600)
            startTrade = None
            endTrade = None
            # exit long
            if pos == 1:
                profit = (O - slippage - prc)*100 # (1 BP = 0.01)
                position = None
                exitTrade = False
                backtestLog.writelines("\n" + str(dt) + " EXIT LONG @ " + str(O - slippage) + "\n                    P/L: " + str(round(profit,1)) + "\n")
                totalLong += 0.5
                if profit > 0:
                    winRecord.append(profit)
                    countWins += 1
                    maxConsecutiveLosses = max(maxConsecutiveLosses,countLosses)
                    countLosses = 0
                    totalLongWins += 1
                elif profit < 0:
                    lossRecord.append(profit)
                    maxConsecutiveWins = max(maxConsecutiveWins,countWins)
                    countWins = 0
                    countLosses += 1
                    totalLongLosses += 1
                else:
                    breakEvenRecord.append(profit)
                    totalLongBE += 1
            # exit short
            elif pos == -1:
                profit = (prc - O - slippage)*100 # (1 BP = 0.01)
                position = None
                exitTrade = False
                backtestLog.writelines("\n" + str(dt) + " EXIT SHORT @ " + str(O + slippage) + "\n                    P/L: " + str(round(profit,1)) + "\n")
                totalShort += 0.5
                if profit > 0:
                    winRecord.append(profit)
                    countWins += 1
                    maxConsecutiveLosses = max(maxConsecutiveLosses,countLosses)
                    countLosses = 0
                    totalShortWins += 1
                elif profit < 0:
                    lossRecord.append(profit)
                    maxConsecutiveWins = max(maxConsecutiveWins,countWins)
                    countWins = 0
                    countLosses += 1
                    totalShortLosses += 1
                else:
                    breakEvenRecord.append(profit)
                    totalShortBE += 1
        # entry signal generator
        if position is None:
            if RSI < enterLong and stoppedOut is False and jump is False:
                if rolls is False or (dt + datetime.timedelta(hours = 1)).day > 14:
                    backtestLog.writelines("\n" + str(dt) + " SIGNAL: RSI = " + str(RSI) )
                enterLongTrade = True
                enterShortTrade = False
            elif RSI > enterShort and stoppedOut is False and jump is False:
                if rolls is False or (dt + datetime.timedelta(hours = 1)).day > 14:
                    backtestLog.writelines("\n" + str(dt) + " SIGNAL: RSI = " + str(RSI) )
                enterLongTrade = False
                enterShortTrade = True
            else:
                enterLongTrade = False
                enterShortTrade = False
        # exit signal generator
        elif position is not None:
            prc = position[1]
            pos = position[2] # 1 if long, -1 if short
            if pos == 1:
                if RSI >= exitLong:
                    exitTrade = True
                    backtestLog.writelines("\n" + str(dt) + " SIGNAL: RSI = " + str(RSI) )
                if (C - prc) <= - stopLoss:
                    exitTrade = True
                    backtestLog.writelines("\n" + str(dt) + " SIGNAL: STOP")
                    if RSI > enterLong and RSI < enterShort:
                        stoppedOut = False
                    else:
                        stoppedOut = True
            elif pos == -1:
                if RSI <= exitShort:
                    exitTrade = True
                    backtestLog.writelines("\n" + str(dt) + " SIGNAL: RSI = " + str(RSI) )
                if (prc - C) <= - stopLoss:
                    exitTrade = True
                    backtestLog.writelines("\n" + str(dt) + " SIGNAL: STOP")
                    if RSI > enterLong and RSI < enterShort:
                        stoppedOut = False
                    else:
                        stoppedOut = True
        finalisedBP = realisedBP[-1][1] + profit
        realisedBP.append([dt,finalisedBP])
    # calculate maximal drawdown & duration
    maximalDrawdown = 0
    maximalDrawdownDuration = 0
    firstDuration = None
    secondDuration = None
    first = 0
    second = 0
    for i in xrange(len(realisedBP)):
        # base
        if i == 0:
            first = realisedBP[0][1]
            second = realisedBP[0][1]
            firstDuration = realisedBP[0][0]
            secondDuration = realisedBP[0][0]
        else:
            # drop
            if second >= realisedBP[i][1]:
                second = realisedBP[i][1]
                secondDuration = realisedBP[i][0]
            else:
                drawdown = first - second
                duration = (secondDuration - firstDuration).total_seconds()/3600
                if maximalDrawdown > drawdown:
                    first = realisedBP[i][1]
                    second = realisedBP[i][1]
                    firstDuration = realisedBP[i][0]
                    secondDuration = realisedBP[i][0]
                elif maximalDrawdown == drawdown:
                    maximalDrawdown = drawdown
                    maximalDrawdownDuration = max(maximalDrawdownDuration,duration)
                    # refresh comparision
                    first = realisedBP[i][1]
                    second = realisedBP[i][1]
                    firstDuration = realisedBP[i][0]
                    secondDuration = realisedBP[i][0]
                else:
                    maximalDrawdownDuration = duration
                    maximalDrawdown = drawdown
                    # refresh comparision
                    first = realisedBP[i][1]
                    second = realisedBP[i][1]
                    firstDuration = realisedBP[i][0]
                    secondDuration = realisedBP[i][0]
            if i == len(realisedBP)-1:
                if maximalDrawdown > drawdown:
                    continue
                elif maximalDrawdown == drawdown:
                    maximalDrawdown = drawdown
                    maximalDrawdownDuration = max(maximalDrawdownDuration,duration)
                else:
                    maximalDrawdown = drawdown
                    maximalDrawdownDuration = duration
    # calculate maximal drawup & duration
    maximalDrawup = 0
    maximalDrawupDuration = 0
    firstDuration = None
    secondDuration = None
    first = 0
    second = 0
    for i in xrange(len(realisedBP)):
        # base
        if i == 0:
            first = realisedBP[0][1]
            second = realisedBP[0][1]
            firstDuration = realisedBP[0][0]
            secondDuration = realisedBP[0][0]
        else:
            # up
            if second <= realisedBP[i][1]:
                second = realisedBP[i][1]
                secondDuration = realisedBP[i][0]
            else:
                drawup = second - first
                duration = (secondDuration - firstDuration).total_seconds()/3600
                if maximalDrawup > drawup:
                    first = realisedBP[i][1]
                    second = realisedBP[i][1]
                    firstDuration = realisedBP[i][0]
                    secondDuration = realisedBP[i][0]
                elif maximalDrawup == drawup:
                    maximalDrawup = drawup
                    maximalDrawupDuration = max(maximalDrawupDuration, duration)
                    # refresh comparision
                    first = realisedBP[i][1]
                    second = realisedBP[i][1]
                    firstDuration = realisedBP[i][0]
                    secondDuration = realisedBP[i][0]
                else:
                    maximalDrawupDuration = duration
                    maximalDrawup = drawup
                    # refresh comparision
                    first = realisedBP[i][1]
                    second = realisedBP[i][1]
                    firstDuration = realisedBP[i][0]
                    secondDuration = realisedBP[i][0]
            if i == len(realisedBP)-1:
                maximalDrawup = max(maximalDrawup,second - first, realisedBP[i][1] - first)
                if maximalDrawup > drawup:
                    continue
                elif maximalDrawdown == drawup:
                    maximalDrawup = drawup
                    maximalDrawupDuration = max(maximalDrawupDuration,duration)
                else:
                    maximalDrawup = drawup
                    maximalDrawupDuration = duration
    # calculate average win,loss,profit per trade
    averageWin = round(np.mean(winRecord),2)
    averageLoss = round(np.mean(lossRecord),2)
    fullRecord = np.array(winRecord + lossRecord + breakEvenRecord)
    averageTrade = round(np.mean(fullRecord),2)
    varianceTrade = round(np.var(fullRecord),2)
    medianTrade = round(np.median(fullRecord),2)
    pWin = round(float(len(winRecord)*100)/len(fullRecord),2)
    pLoss = round(float(len(lossRecord)*100)/len(fullRecord),2)
    pBreakEven = round(float(len(breakEvenRecord)*100)/len(fullRecord),2)
    largestWin = np.max(fullRecord)
    largestLoss = np.min(fullRecord)
    backtestLog.write("\n\nAssumed slippage of " + str(slippage*100) + " Basis Points per side." )
    backtestLog.write("\n\nTrade Profit/Loss Statistics (in Basis Points):")
    backtestLog.write("\nTotal Realised Basis Points, " + str(round(finalisedBP,1)))
    backtestLog.write("\nTrade P/L Mean, " + str(averageTrade) +
                      "\nTrade P/L Variance, " + str(varianceTrade) +
                      "\nTrade P/L Median, " + str(medianTrade))
    backtestLog.write("\n\nWins & Losses (in Basis Points):")
    backtestLog.write("\nWin/Loss Mean, " + str(averageWin) + "/" + str(averageLoss))
    backtestLog.write("\nWin/Loss Variance, " + str(round(np.var(winRecord),2)) + "/" + str(round(np.var(lossRecord),2)))
    backtestLog.write("\nWin/Loss Median, " + str(np.median(winRecord)) + "/" + str(np.median(lossRecord)))
    backtestLog.write("\nLargest Win/Loss, " + str(largestWin) + "/" + str(largestLoss))
    backtestLog.write("\n\nTrade Statistics:")
    backtestLog.write("\nTotal Trades, " + str(totalLong + totalShort))
    backtestLog.write("\nWin/Loss/Break-Even, " +
                      str(totalLongWins+totalShortWins) + "/" +
                      str(totalLongLosses+totalShortLosses) + "/" +
                      str(totalLongBE+totalShortBE))
    backtestLog.write("\nPercentage Win/Loss/Break-Even, " +
                      str(pWin) + "% / " +
                      str(pLoss) + "% / " +
                      str(pBreakEven) + "%")
    backtestLog.write("\nAverage Duration of Trade, " + str(round(np.mean(timeRecord),2)) + " hours")
    pLongWin = round(totalLongWins/totalLong*100,2)
    pLongLoss = round(totalLongLosses/totalLong*100,2)
    pLongBreakEven = round(totalLongBE/totalLong*100,2)
    backtestLog.write("\n\nLong Trade Statistics:\nTotal Long Trades, " + str(totalLong) +
                      "\nWin/Loss/Break-Even, " + str(totalLongWins) + "/" + str(totalLongLosses) + "/" + str(totalLongBE) +
                      "\nPercentage Win/Loss/Break-Even, " +
                      str(pLongWin) + "% / " +
                      str(pLongLoss) + "% / " +
                      str(pLongBreakEven) + "%")
    pShortWin = round(totalShortWins/totalShort*100,2)
    pShortLoss = round(totalShortLosses/totalShort*100,2)
    pShortBreakEven = round(totalShortBE/totalShort*100,2)
    backtestLog.write("\n\nShort Trade Statistics:\nTotal Short Trades, " + str(totalShort) +
                      "\nWin/Loss/Break-Even, " + str(totalShortWins) + "/" + str(totalShortLosses) + "/" + str(totalShortBE) +
                      "\nPercentage Win/Loss/Break-Even, " +
                      str(pShortWin) + "% / " +
                      str(pShortLoss) + "% / " +
                      str(pShortBreakEven) + "%")
    backtestLog.write("\n\nPeak-Trough Statistics (in Basis Points):\nMaximal Drawdown (Duration), -" +
                      str(maximalDrawdown) + " (" + str(round(maximalDrawdownDuration,2)) + " hours)" +
                      "\nMaximal Draw-up (Duration), " +
                      str(maximalDrawup) + " (" + str(round(maximalDrawupDuration,2)) + " hours)" )
    backtestLog.write("\nMaximum Consecutive Wins/Losses, " + str(maxConsecutiveWins) + "/" + str(maxConsecutiveLosses))
    backtestLog.write("\n\nTime Elapsed: " + str(time.time() - r) + " seconds")
    backtestLog.close()
    df = pd.DataFrame(realisedBP,columns = ["dt","bp"])
    plt.plot(df["dt"],df["bp"],colour)
    plt.grid(b = True, which = "major", color = "gray")
    plt.title("Walk-Forward Performance (in Basis Points) versus Time \n")
    if returnList:
        return realisedBP
    else:
        return realisedBP[-1]

# trend following backtest: 1 BP = 100
def trendFollowing(HXS, slippage = 0, printTrade = False):
    realisedBP = [[HXS[0][0],0]]
    unrealisedBP = [[HXS[0][0],0]]
    position = None
    for i in xrange(len(HXS)):
        # record first row details
        if i == 0:
            continue
        else:
            # print(str(position) + ": " + str(realisedBP[-1][1]) + "/" + str(unrealisedBP[-1][1]))
            row = HXS[i]
            dt = row[0]
            #O = row[1]
            #H = row[2]
            #L = row[3]
            previousC = HXS[i-1][4]
            C = row[4]
            previousMA = HXS[i-1][5]
            MA = row[5]
            # case for self calculated MA
            if MA is np.NaN or previousMA is np.NaN:
                continue
            # check signal for entry
            if position is None:
                # long if MA crosses C from above
                if (MA < C and previousMA >= previousC):
                    # take C price
                    position = [dt, C + slippage, 1]
                    if printTrade:
                        print(str(dt) + ", Enter Long @ " + str(C))
                        print("C:" + str(previousC) + " previousC:" + str(C))
                        print("MA:" + str(MA) + " previousMA:" + str(previousMA))
                # short if MA crosses C from below
                elif (MA > C and previousMA <= previousC):
                    # take C price
                    position = [dt, C + slippage, -1]
                    if printTrade:
                        print(str(dt) + ", Enter Short @ " + str(C))
                        print("C:" + str(previousC) + " previousC:" + str(C))
                        print("MA:" + str(MA) + " previousMA:" + str(previousMA))
                else:
                    position = None
                # record basis points trail
                realisedBP.append([dt, realisedBP[-1][1]])
                unrealisedBP.append([dt, unrealisedBP[-1][1]])
            # check signal for exit long; enter short
            elif position[2] == 1:
                # exit long & enter short if MA crosses C from below
                if (MA > C and previousMA <= previousC):
                    # close long trade
                    profit = (C - position[1])*0.01 # 100 = 1 bp
                    basisPoints = realisedBP[-1][1] + profit
                    realisedBP.append([dt, basisPoints])
                    unrealisedBP.append([dt, basisPoints])
                    if printTrade:
                        print(str(dt) + ", Exit Long @ " + str(C) + " ... Profit: " + str(profit))
                        print("C:" + str(C) + " previousC:" + str(previousC))
                        print("MA:" + str(MA) + " previousMA:" + str(previousMA))
                    # enter short trade
                    position = [dt, C + slippage, -1]
                    if printTrade:
                        print(str(dt) + ", Enter Short @ " + str(C))
                        print("C:" + str(C) + " previousC:" + str(previousC))
                        print("MA:" + str(MA) + " previousMA:" + str(previousMA))
                else:
                    # record basis point trail
                    profit = (C - position[1])*0.01 # 100 = 1 bp
                    basisPoints = realisedBP[-1][1] + profit
                    realisedBP.append([dt, realisedBP[-1][1]])
                    unrealisedBP.append([dt, basisPoints])
            elif position[2] == -1:
                # exit short & enter long if MA crosses C from above
                if (MA < C and previousMA >= previousC):
                    # close short trade
                    profit = (position[1] - C)*0.01 # 100 = 1 bp
                    basisPoints = realisedBP[-1][1] + profit
                    realisedBP.append([dt, basisPoints])
                    unrealisedBP.append([dt, basisPoints])
                    if printTrade:
                        print(str(dt) + ", Exit Short @ " + str(C) + " ... Profit: " + str(profit))
                        print("C:" + str(C) + " previousC:" + str(previousC))
                        print("MA:" + str(MA) + " previousMA:" + str(previousMA))
                    # enter long trade
                    position = [dt, C + slippage, 1]
                    if printTrade:
                        print(str(dt) + ", Enter Long @ " + str(C))
                        print("C:" + str(C) + " previousC:" + str(previousC))
                        print("MA:" + str(MA) + " previousMA:" + str(previousMA))
                else:
                    # record basis point trail
                    profit  = (position[1] - C)*0.01 # 100 = 1 bp
                    basisPoints = realisedBP[-1][1] + profit
                    realisedBP.append([dt, realisedBP[-1][1]])
                    unrealisedBP.append([dt, basisPoints])
    return [unrealisedBP,realisedBP]

# given data in time series list, change column [5] entries
def changeSMA(data,period):
    # append close price
    close = [i[4] for i in data]
    # calculate SMA based on period argument
    close = pd.DataFrame(close)
    SMA = pd.rolling_mean(close, period)
    # replace indicate/SMA values
    i = 0
    for value in SMA[0]:
        data[i][5] = value
        i += 1


# optimisation of trend following based on realised basis points
def optimiseTrendFollowing(HXS, movingAverageLower = 100, movingAverageUpper = 200,
                           increment = 1, display = True):
    # just to count total cases
    totalCases = int((movingAverageUpper - movingAverageLower)/increment + 1)
    # base case, set counter and SMA
    case = 0
    movingAverage = movingAverageLower
    changeSMA(HXS,movingAverage)
    # start optimising
    parameters = []
    bestRealisedBasisPoints = 0
    t = time.time()
    while movingAverage <= movingAverageUpper:
        result = trendFollowing(HXS)[1]
        if result[-1][-1] > bestRealisedBasisPoints:
            bestRealisedBasisPoints = result[-1][-1]
            parameters = [movingAverage, bestRealisedBasisPoints]
        del result
        case += 1
        if display is True and case % 100 == 0:
            print("Completed " + str(case) + " of " + str(totalCases) + ". Time Elapsed: " + str(time.time() - t) + "s")
        # increment moving average
        movingAverage += increment
        changeSMA(HXS,movingAverage)
    if display is True:
        print("Completed " + str(case) + " of " + str(totalCases) + ". Time Elapsed: " + str(time.time() - t) + "s")
    return parameters


# backtest function (minimal details, quicker), modified RSI "cut through" conditions ...
def backtestCutRSI(timeSeries, enterLong = 35, exitLong = 53,
             enterShort = 60, exitShort = 49, stopLoss = 0.04,
             record = False, slippage = 0.005, returnList = False):
    # write trade record option
    if record is True:
        r = time.time()
        backtestLog = open("backtestCutRSI_Log","w")
        backtestLog.write("BACKTEST TRADE LOG: " + str(timeSeries[0][0].date()) + " to " + str(timeSeries[-1][0].date()) + "\n")
    realisedBP = [[timeSeries[0][0], 0]]
    position = None # or [ dt, price (entry), +/- 1] if filled
    stoppedOut = False
    exitTrade = False
    enterLongTrade = False
    enterShortTrade = False
    for i in xrange(len(timeSeries)):
        profit = 0
        row = timeSeries[i]
        dt = row[0]
        O = row[1]
        H = row[2]
        L = row[3]
        C = row[4]
        RSI = row[5]
        # consider past RSI
        previousRSI = timeSeries[max(0,i-1)][5]
        # look out for jumps
        jump = False
        if i > 0:
            rowPrevious = timeSeries[i-1]
            if L >= rowPrevious[2] + 0.02 or H <= rowPrevious[3] - 0.02:
                jump = True
        # look out for futures rollover period (no entry during rolls)
        rolls = False
        if dt.month in [3,6,9,12]:
            if 8 <= dt.day and dt.day <= 14:
                rolls = True
        # refresh stopping out history if RSI crosses out
        if stoppedOut is True:
            if RSI > enterLong and RSI < enterShort:
                stoppedOut = False
            else:
                stoppedOut = False
        # long entry
        if (enterLongTrade is True
        and position is None and rolls is False and jump is False and stoppedOut is False):
            position = [dt, O + slippage, 1]
            enterLongTrade = False
            if record is True:
                backtestLog.writelines("\n" + str(position[0]) + " ENTER LONG @ " + str(position[1]))
        # short entry
        if (enterShortTrade is True
        and position is None and rolls is False and jump is False and stoppedOut is False):
            position = [dt, O - slippage, -1]
            enterShortTrade = False
            if record is True:
                backtestLog.writelines("\n" + str(position[0]) + " ENTER SHORT @ " + str(position[1]))
        # exit trade
        if exitTrade is True:
            prc = position[1]
            pos = position[2]
            # exit long
            if pos == 1:
                profit = (O - slippage - prc)*100 # (1 BP = 0.01)
                position = None
                exitTrade = False
                if record is True:
                    backtestLog.writelines("\n" + str(dt) + " EXIT LONG @ " + str(O - slippage) + "\n                    P/L: " + str(round(profit,1)) + "\n")
            # exit short
            elif pos == -1:
                profit = (prc - O - slippage)*100 # (1 BP = 0.01)
                position = None
                exitTrade = False
                if record is True:
                    backtestLog.writelines("\n" + str(dt) + " EXIT SHORT @ " + str(O + slippage) + "\n                    P/L: " + str(round(profit,1)) + "\n")
        # entry signal generator
        if position is None:
            # long when RSI cuts through enterLong from below
            if previousRSI < enterLong and RSI > enterLong and stoppedOut is False and jump is False:
                if rolls is False or (dt + datetime.timedelta(hours = 1)).day > 14:
                    backtestLog.writelines("\n" + str(dt) + " SIGNAL: RSI = " + str(RSI) )
                enterLongTrade = True
                enterShortTrade = False
            # short when RSI cuts through enterShort from above
            elif previousRSI > enterShort and RSI < enterShort and stoppedOut is False and jump is False:
                if rolls is False or (dt + datetime.timedelta(hours = 1)).day > 14:
                    backtestLog.writelines("\n" + str(dt) + " SIGNAL: RSI = " + str(RSI) )
                enterLongTrade = False
                enterShortTrade = True
            else:
                enterLongTrade = False
                enterShortTrade = False
        # exit signal generator
        elif position is not None:
            prc = position[1]
            pos = position[2] # 1 if long, -1 if short
            if pos == 1:
                if RSI >= exitLong:
                    exitTrade = True
                if (C - prc) <= - stopLoss:
                    exitTrade = True
                    if RSI > enterLong and RSI < enterShort:
                        stoppedOut = False
                    else:
                        stoppedOut = False
            elif pos == -1:
                if RSI <= exitShort:
                    exitTrade = True
                if (prc - C) <= - stopLoss:
                    exitTrade = True
                    if RSI > enterLong and RSI < enterShort:
                        stoppedOut = False
                    else:
                        stoppedOut = False
        finalisedBP = realisedBP[-1][1] + profit
        realisedBP.append([dt,finalisedBP])
    if record is True:
        backtestLog.write("\nRealised Basis Points: " + str(round(finalisedBP,1)) + "\n")
        backtestLog.write("\nTime Elapsed: " + str(time.time() - r ) + " seconds")
        backtestLog.close()
    # default output
    if returnList is False:
        return finalisedBP
    # or output entire history
    else:
        return realisedBP


# backtest function (detailed statistics report with plots, slower), modified RSI "cut through" condition ...
def backtestCutRSIDetailedPlot(timeSeries, enterLong = 35, exitLong = 53,
                         enterShort = 60, exitShort = 49, stopLoss = 0.04,
                         slippage = 0.005, colour = "blue", returnList = False):
    # always write trade record
    r = time.time()
    backtestLog = open("backtestLogCutRSI_Detailed","w")
    backtestLog.write("BACKTEST TRADE LOG:\nfrom " + str(timeSeries[0][0].date()) + " to " + str(timeSeries[-1][0].date()) + "\n")
    realisedBP = [[timeSeries[0][0], 0]]
    position = None # or [ dt, price (entry), +/- 1] if filled
    stoppedOut = False
    exitTrade = False
    enterLongTrade = False
    enterShortTrade = False
    # statistics for reporting
    totalShort, totalShortWins, totalShortLosses, totalShortBE = 0, 0, 0, 0
    totalLong, totalLongWins, totalLongLosses, totalLongBE = 0, 0, 0, 0
    maxConsecutiveWins, maxConsecutiveLosses = 0, 0
    countWins, countLosses = 0, 0
    winRecord = []
    lossRecord = []
    breakEvenRecord = []
    # for calculating duration of trade
    timeRecord = []
    startTrade = None
    endTrade = None
    for i in xrange(len(timeSeries)):
        profit = 0
        row = timeSeries[i]
        dt = row[0]
        O = row[1]
        H = row[2]
        L = row[3]
        C = row[4]
        RSI = row[5]
        previousRSI = timeSeries[max(0,i-1)][5]
        # look out for jumps
        jump = False
        if i > 0:
            rowPrevious = timeSeries[i-1]
            if L >= rowPrevious[2] + 0.02 or H <= rowPrevious[3] - 0.02:
                jump = True
                #backtestLog.write("\n" + str(dt) + " JUMP")
        # look out for futures rollover period (no entry during rolls)
        rolls = False
        if dt.month in [3,6,9,12]:
            if 8 <= dt.day and dt.day <= 14:
                rolls = True
        # refresh stopping out history if RSI crosses out
        if stoppedOut is True:
            if RSI > enterLong and RSI < enterShort:
                stoppedOut = False
            else:
                stoppedOut = False
        # long entry
        if (enterLongTrade is True
        and position is None and rolls is False and jump is False and stoppedOut is False):
            startTrade = dt
            position = [dt, O + slippage, 1]
            enterLongTrade = False
            backtestLog.writelines("\n" + str(position[0]) + " ENTER LONG @ " + str(position[1]))
            totalLong += 0.5
        # short entry
        if (enterShortTrade is True
        and position is None and rolls is False and jump is False and stoppedOut is False):
            startTrade = dt
            position = [dt, O - slippage, -1]
            enterShortTrade = False
            backtestLog.writelines("\n" + str(position[0]) + " ENTER SHORT @ " + str(position[1]))
            totalShort += 0.5
        # exit trade
        if exitTrade is True:
            prc = position[1]
            pos = position[2]
            endTrade = dt
            timeRecord.append((endTrade - startTrade).total_seconds()/3600)
            startTrade = None
            endTrade = None
            # exit long
            if pos == 1:
                profit = (O - slippage - prc)*100 # (1 BP = 0.01)
                position = None
                exitTrade = False
                backtestLog.writelines("\n" + str(dt) + " EXIT LONG @ " + str(O - slippage) + "\n                    P/L: " + str(round(profit,1)) + "\n")
                totalLong += 0.5
                if profit > 0:
                    winRecord.append(profit)
                    countWins += 1
                    maxConsecutiveLosses = max(maxConsecutiveLosses,countLosses)
                    countLosses = 0
                    totalLongWins += 1
                elif profit < 0:
                    lossRecord.append(profit)
                    maxConsecutiveWins = max(maxConsecutiveWins,countWins)
                    countWins = 0
                    countLosses += 1
                    totalLongLosses += 1
                else:
                    breakEvenRecord.append(profit)
                    totalLongBE += 1
            # exit short
            elif pos == -1:
                profit = (prc - O - slippage)*100 # (1 BP = 0.01)
                position = None
                exitTrade = False
                backtestLog.writelines("\n" + str(dt) + " EXIT SHORT @ " + str(O + slippage) + "\n                    P/L: " + str(round(profit,1)) + "\n")
                totalShort += 0.5
                if profit > 0:
                    winRecord.append(profit)
                    countWins += 1
                    maxConsecutiveLosses = max(maxConsecutiveLosses,countLosses)
                    countLosses = 0
                    totalShortWins += 1
                elif profit < 0:
                    lossRecord.append(profit)
                    maxConsecutiveWins = max(maxConsecutiveWins,countWins)
                    countWins = 0
                    countLosses += 1
                    totalShortLosses += 1
                else:
                    breakEvenRecord.append(profit)
                    totalShortBE += 1
        # entry signal generator
        if position is None:
            # long when RSI cuts through enterLong from below
            if previousRSI < enterLong and RSI > enterLong and stoppedOut is False and jump is False:
                if rolls is False or (dt + datetime.timedelta(hours = 1)).day > 14:
                    backtestLog.writelines("\n" + str(dt) + " SIGNAL: RSI = " + str(RSI) )
                enterLongTrade = True
                enterShortTrade = False
            # short when RSI cuts through enterShort from above
            elif previousRSI > enterShort and RSI < enterShort and stoppedOut is False and jump is False:
                if rolls is False or (dt + datetime.timedelta(hours = 1)).day > 14:
                    backtestLog.writelines("\n" + str(dt) + " SIGNAL: RSI = " + str(RSI) )
                enterLongTrade = False
                enterShortTrade = True
            else:
                enterLongTrade = False
                enterShortTrade = False
        # exit signal generator
        elif position is not None:
            prc = position[1]
            pos = position[2] # 1 if long, -1 if short
            if pos == 1:
                if RSI >= exitLong:
                    exitTrade = True
                    backtestLog.writelines("\n" + str(dt) + " SIGNAL: RSI = " + str(RSI) )
                if (C - prc) <= - stopLoss:
                    exitTrade = True
                    backtestLog.writelines("\n" + str(dt) + " SIGNAL: STOP")
                    if RSI > enterLong and RSI < enterShort:
                        stoppedOut = False
                    else:
                        stoppedOut = False
            elif pos == -1:
                if RSI <= exitShort:
                    exitTrade = True
                    backtestLog.writelines("\n" + str(dt) + " SIGNAL: RSI = " + str(RSI) )
                if (prc - C) <= - stopLoss:
                    exitTrade = True
                    backtestLog.writelines("\n" + str(dt) + " SIGNAL: STOP")
                    if RSI > enterLong and RSI < enterShort:
                        stoppedOut = False
                    else:
                        stoppedOut = False
        finalisedBP = realisedBP[-1][1] + profit
        realisedBP.append([dt,finalisedBP])
    # calculate maximal drawdown & duration
    maximalDrawdown = 0
    maximalDrawdownDuration = 0
    firstDuration = None
    secondDuration = None
    first = 0
    second = 0
    for i in xrange(len(realisedBP)):
        # base
        if i == 0:
            first = realisedBP[0][1]
            second = realisedBP[0][1]
            firstDuration = realisedBP[0][0]
            secondDuration = realisedBP[0][0]
        else:
            # drop
            if second >= realisedBP[i][1]:
                second = realisedBP[i][1]
                secondDuration = realisedBP[i][0]
            else:
                drawdown = first - second
                duration = (secondDuration - firstDuration).total_seconds()/3600
                if maximalDrawdown > drawdown:
                    first = realisedBP[i][1]
                    second = realisedBP[i][1]
                    firstDuration = realisedBP[i][0]
                    secondDuration = realisedBP[i][0]
                elif maximalDrawdown == drawdown:
                    maximalDrawdown = drawdown
                    maximalDrawdownDuration = max(maximalDrawdownDuration,duration)
                    # refresh comparision
                    first = realisedBP[i][1]
                    second = realisedBP[i][1]
                    firstDuration = realisedBP[i][0]
                    secondDuration = realisedBP[i][0]
                else:
                    maximalDrawdownDuration = duration
                    maximalDrawdown = drawdown
                    # refresh comparision
                    first = realisedBP[i][1]
                    second = realisedBP[i][1]
                    firstDuration = realisedBP[i][0]
                    secondDuration = realisedBP[i][0]
            if i == len(realisedBP)-1:
                if maximalDrawdown > drawdown:
                    continue
                elif maximalDrawdown == drawdown:
                    maximalDrawdown = drawdown
                    maximalDrawdownDuration = max(maximalDrawdownDuration,duration)
                else:
                    maximalDrawdown = drawdown
                    maximalDrawdownDuration = duration
    # calculate maximal drawup & duration
    maximalDrawup = 0
    maximalDrawupDuration = 0
    firstDuration = None
    secondDuration = None
    first = 0
    second = 0
    for i in xrange(len(realisedBP)):
        # base
        if i == 0:
            first = realisedBP[0][1]
            second = realisedBP[0][1]
            firstDuration = realisedBP[0][0]
            secondDuration = realisedBP[0][0]
        else:
            # up
            if second <= realisedBP[i][1]:
                second = realisedBP[i][1]
                secondDuration = realisedBP[i][0]
            else:
                drawup = second - first
                duration = (secondDuration - firstDuration).total_seconds()/3600
                if maximalDrawup > drawup:
                    first = realisedBP[i][1]
                    second = realisedBP[i][1]
                    firstDuration = realisedBP[i][0]
                    secondDuration = realisedBP[i][0]
                elif maximalDrawup == drawup:
                    maximalDrawup = drawup
                    maximalDrawupDuration = max(maximalDrawupDuration, duration)
                    # refresh comparision
                    first = realisedBP[i][1]
                    second = realisedBP[i][1]
                    firstDuration = realisedBP[i][0]
                    secondDuration = realisedBP[i][0]
                else:
                    maximalDrawupDuration = duration
                    maximalDrawup = drawup
                    # refresh comparision
                    first = realisedBP[i][1]
                    second = realisedBP[i][1]
                    firstDuration = realisedBP[i][0]
                    secondDuration = realisedBP[i][0]
            if i == len(realisedBP)-1:
                maximalDrawup = max(maximalDrawup,second - first, realisedBP[i][1] - first)
                if maximalDrawup > drawup:
                    continue
                elif maximalDrawdown == drawup:
                    maximalDrawup = drawup
                    maximalDrawupDuration = max(maximalDrawupDuration,duration)
                else:
                    maximalDrawup = drawup
                    maximalDrawupDuration = duration
    # calculate average win,loss,profit per trade
    averageWin = round(np.mean(winRecord),2)
    averageLoss = round(np.mean(lossRecord),2)
    fullRecord = np.array(winRecord + lossRecord + breakEvenRecord)
    averageTrade = round(np.mean(fullRecord),2)
    varianceTrade = round(np.var(fullRecord),2)
    medianTrade = round(np.median(fullRecord),2)
    pWin = round(float(len(winRecord)*100)/len(fullRecord),2)
    pLoss = round(float(len(lossRecord)*100)/len(fullRecord),2)
    pBreakEven = round(float(len(breakEvenRecord)*100)/len(fullRecord),2)
    largestWin = np.max(fullRecord)
    largestLoss = np.min(fullRecord)
    backtestLog.write("\nParameters Used: "  + str([enterLong, exitLong, enterShort,exitShort,stopLoss*100]))
    backtestLog.write("\nAssumed slippage of " + str(slippage*100) + " Basis Points per side." )
    backtestLog.write("\n\nTrade Profit/Loss Statistics (in Basis Points):")
    backtestLog.write("\nTotal Realised Basis Points, " + str(round(finalisedBP,1)))
    backtestLog.write("\nTrade P/L Mean, " + str(averageTrade) +
                      "\nTrade P/L Variance, " + str(varianceTrade) +
                      "\nTrade P/L Median, " + str(medianTrade))
    backtestLog.write("\n\nWins & Losses (in Basis Points):")
    backtestLog.write("\nWin/Loss Mean, " + str(averageWin) + "/" + str(averageLoss))
    backtestLog.write("\nWin/Loss Variance, " + str(round(np.var(winRecord),2)) + "/" + str(round(np.var(lossRecord),2)))
    backtestLog.write("\nWin/Loss Median, " + str(np.median(winRecord)) + "/" + str(np.median(lossRecord)))
    backtestLog.write("\nLargest Win/Loss, " + str(largestWin) + "/" + str(largestLoss))
    backtestLog.write("\n\nTrade Statistics:")
    backtestLog.write("\nTotal Trades, " + str(totalLong + totalShort))
    backtestLog.write("\nWin/Loss/Break-Even, " +
                      str(totalLongWins+totalShortWins) + "/" +
                      str(totalLongLosses+totalShortLosses) + "/" +
                      str(totalLongBE+totalShortBE))
    backtestLog.write("\nPercentage Win/Loss/Break-Even, " +
                      str(pWin) + "% / " +
                      str(pLoss) + "% / " +
                      str(pBreakEven) + "%")
    backtestLog.write("\nAverage Duration of Trade, " + str(round(np.mean(timeRecord),2)) + " hours")
    pLongWin = round(totalLongWins/totalLong*100,2)
    pLongLoss = round(totalLongLosses/totalLong*100,2)
    pLongBreakEven = round(totalLongBE/totalLong*100,2)
    backtestLog.write("\n\nLong Trade Statistics:\nTotal Long Trades, " + str(totalLong) +
                      "\nWin/Loss/Break-Even, " + str(totalLongWins) + "/" + str(totalLongLosses) + "/" + str(totalLongBE) +
                      "\nPercentage Win/Loss/Break-Even, " +
                      str(pLongWin) + "% / " +
                      str(pLongLoss) + "% / " +
                      str(pLongBreakEven) + "%")
    pShortWin = round(totalShortWins/totalShort*100,2)
    pShortLoss = round(totalShortLosses/totalShort*100,2)
    pShortBreakEven = round(totalShortBE/totalShort*100,2)
    backtestLog.write("\n\nShort Trade Statistics:\nTotal Short Trades, " + str(totalShort) +
                      "\nWin/Loss/Break-Even, " + str(totalShortWins) + "/" + str(totalShortLosses) + "/" + str(totalShortBE) +
                      "\nPercentage Win/Loss/Break-Even, " +
                      str(pShortWin) + "% / " +
                      str(pShortLoss) + "% / " +
                      str(pShortBreakEven) + "%")
    backtestLog.write("\n\nPeak-Trough Statistics (in Basis Points):\nMaximal Drawdown (Duration), -" +
                      str(maximalDrawdown) + " (" + str(round(maximalDrawdownDuration,2)) + " hours)" +
                      "\nMaximal Draw-up (Duration), " +
                      str(maximalDrawup) + " (" + str(round(maximalDrawupDuration,2)) + " hours)" )
    backtestLog.write("\nMaximum Consecutive Wins/Losses, " + str(maxConsecutiveWins) + "/" + str(maxConsecutiveLosses))

    backtestLog.write("\n\nTime Elapsed: " + str(time.time() - r) + " seconds")
    backtestLog.close()
    df = pd.DataFrame(realisedBP,columns = ["dt","bp"])
    plt.plot(df["dt"],df["bp"],colour)
    plt.grid(b = True, which = "major", color = "gray")
    plt.title("Performance (in Basis Points) versus Time \n")
    if returnList:
        return realisedBP
    else:
        return realisedBP[-1]


# open HXS
fileHXS = "HXS_Cleaned.csv"
# read HXS
csvHXS = csv.reader(open(fileHXS,"r"))
# plaxe HXS into list
HXS = createTimeSeries(csvHXS)
df = pd.DataFrame(HXS,columns = ["dt","O","H","L","C","MA"])

# open file
fileName = "CurveTensTens_Cleaned.csv"
# read file
csvFile = csv.reader(open(fileName,"r"))
# place data into list
timeSeries = createTimeSeries(csvFile)

""" # walk forward stuff
walkforward(timeSeries,optimisationPeriod=24,testPeriod=1,target="sortino")
print("FINISHED WALK FORWARD OPTIMISATION.")
""" #For manipulating text log
a = open("WFL.txt","r")
outputDetails = [[24,1]]
for i in a:
    if "Finished" in i:
        enterLong = float(i[24:26])
        exitLong = float(i[28:30])
        enterShort = float(i[32:34])
        exitShort = float(i[36:38])
        stopLoss = float(i[40:44])
        # print(i)
        # print(str(enterLong) + " " + str(exitLong) + " " + str(enterShort) + " " + str(exitShort) + " " + str(stopLoss))
        outputDetails.append([enterLong, exitLong, enterShort, exitShort, stopLoss])

backtestDetailedPlotWF(timeSeries,outputDetails,colour = "blue")

realisedPerformanceCTT = backtest(timeSeries,
                                  enterLong = 31, exitLong = 53,
                                  enterShort = 60, exitShort = 41,
                                  stopLoss = 0.05,
                                  record = False, slippage = 0.005, returnList = True)
DFrealisedPerformanceCTT = pd.DataFrame(realisedPerformanceCTT,columns = ["dt","bp"])

plt.plot(DFrealisedPerformanceCTT["dt"],DFrealisedPerformanceCTT["bp"],'brown')
# formatting
leg = plt.legend(["Walk Forward","Standard"],
                  loc='best')
for legobj in leg.legendHandles:
    legobj.set_linewidth(5.0)
plt.grid(b = True, which = "major", color = "gray")
#plt.title("Performance vs. Time")
plt.xlabel("Time")
plt.ylabel("Basis Points")
plt.title("Walk Forward Performance, Standard Performance")
plt.savefig('ctt_chart.pdf', format='pdf')


"""
x = datetime.datetime(2014, 7, 25)
y = datetime.datetime(2015, 1, 25)
z = sliceTimeSeries(timeSeries, x, y)
# e.g. walk forward with optimisation period of 24 months, testing period of 1 month
# + present details
#backtestDetailedPlotWF(timeSeries,walkforward(timeSeries,24,1,target="sortino"))

print("Finished loading CurveTensTens.py module...")
print("Generating backtest results...")

# run backtest based on 2012-2015 best bp optimisation:
realisedPerformanceCTT = backtest(timeSeries,
                                  enterLong = 31, exitLong = 53,
                                  enterShort = 60, exitShort = 41,
                                  stopLoss = 0.05,
                                  record = False, slippage = 0.005, returnList = True)
DFrealisedPerformanceCTT = pd.DataFrame(realisedPerformanceCTT,columns = ["dt","bp"])
# run 200-SMA trendfollowing backtest
print("TREND FOLLOWING TRADE LOG:")
trendFollowingResults = trendFollowing(HXS,printTrade = True)
unrealisedPerformanceHXS = trendFollowingResults[0]
realisedPerformanceHXS = trendFollowingResults[1]
DFunrealisedPerformanceHXS = pd.DataFrame(unrealisedPerformanceHXS, columns = ["dt","bp"])
DFrealisedPerformanceHXS = pd.DataFrame(realisedPerformanceHXS, columns = ["dt","bp"])
# aggregate performance: realised CTT + unrealised Trend Following
aggregatePerformance = []
j = 0
for i in range(len(realisedPerformanceCTT)):
    dt = realisedPerformanceCTT[i][0]
    bp = realisedPerformanceCTT[i][1]
    while unrealisedPerformanceHXS[j][0].date() != dt.date():
        j += 1
        if j >= len(unrealisedPerformanceHXS):
            j = 0
            bp = aggregatePerformance[-1][1]
            break
    bp += unrealisedPerformanceHXS[j][1]
    aggregatePerformance.append([dt,bp])
DFaggregatePerformance = pd.DataFrame(aggregatePerformance,columns = ["dt","bp"])
# aggregate plot
print("Plotting results...")

plt.plot(DFaggregatePerformance["dt"],DFaggregatePerformance["bp"])
# individual plots
plt.plot(DFrealisedPerformanceCTT["dt"],DFrealisedPerformanceCTT["bp"],'brown')
plt.plot(DFunrealisedPerformanceHXS["dt"],DFunrealisedPerformanceHXS["bp"],'gray')
plt.plot(DFrealisedPerformanceHXS["dt"],DFrealisedPerformanceHXS["bp"],'black')
# formatting
leg = plt.legend(["Aggregate Perforamce","Curve Tens Tens Strategy","Trend Following (Unrealised)","Trend Following (Realised)"],
                  loc='best')
for legobj in leg.legendHandles:
    legobj.set_linewidth(5.0)
plt.grid(b = True, which = "major", color = "gray")
#plt.title("Performance vs. Time")
plt.xlabel("Time")
plt.ylabel("Basis Points")
"""

""" for visualising HXS close and moving average
HXS_DF = pd.DataFrame(HXS, columns = ["dt","O","H","L","C","SMA"])

plt.plot(HXS_DF["dt"],HXS_DF["C"])
plt.plot(HXS_DF["dt"],HXS_DF["SMA"])
plt.grid(b = True, which = "major", color = "gray")
"""
print("Done.")