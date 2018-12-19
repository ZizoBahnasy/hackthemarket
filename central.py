from pandas_datareader import data as pdr
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

from math import log

import collections
import csv
import datetime
from datetime import date, timedelta
import numpy as np

from sklearn import metrics
from sklearn.naive_bayes import GaussianNB

import random


import fix_yahoo_finance as yf, numpy as np
yf.pdr_override() # <== that's all it takes :-)

pd.options.mode.chained_assignment = None
US_BUSINESS_DAY = CustomBusinessDay(calendar=USFederalHolidayCalendar())

ticker = "BABA"

# deltaList = ["Very Much Up", "Up", "Neutral", "Down", "Very Much Down"]
deltaList = ["Up", "Neutral", "Down"]

def ema(stock, period, code):
    if code == "EMA":
        if period < len(stock.index):
            for i in range(period, len(stock.index)):

                closingPrices = [stock['Adj Close'][i - j] for j in range(period, 0, -1)]

                sma = sum(closingPrices)/period

                multiplier = 2.0/(period + 1)
                stock['EMA' + str(period)][i] = sma

                if i != period:
                    stock['EMA' + str(period)][i] = (stock['Adj Close'][i] - stock['EMA' + str(period)][i - 1]) * multiplier + stock['EMA' + str(period)][i - 1]
            return stock
        else:
            for i in range(len(stock.index)):
                closingPrices = [stock['Adj Close'][j] for j in range(len(stock.index))]
                sma = sum(closingPrices)/len(stock.index)
                stock['EMA' + str(period)][i] = sma
            return stock
    else:
        if period < len(stock.index):
            for i in range(period, len(stock.index)):
                macdList = [stock['MACD'][i - j] for j in range(period, 0, -1)]
                sma = sum(macdList)/period
                multiplier = 2.0/(period + 1)
                stock['MACD Signal'][i] = sma
                if i != period:
                    if stock['EMA12'][i] != 0 and stock['EMA26'][i] != 0:
                        stock['MACD Signal'][i] = ((stock['EMA12'][i] - stock['EMA26'][i]) - stock['MACD Signal'][i - 1]) * multiplier + stock['MACD Signal'][i - 1]
                    else:
                        stock['MACD Signal'][i] = 0

                if stock['MACD'][i] < stock['MACD Signal'][i] and stock['MACD'][i - 1] < stock['MACD Signal'][i - 1]:
                    stock['MACD Trend'][i] = 1.0
                if stock['MACD'][i] < stock['MACD Signal'][i] and stock['MACD'][i - 1] > stock['MACD Signal'][i - 1]:
                    stock['MACD Trend'][i] = 2.0
                if stock['MACD'][i] > stock['MACD Signal'][i] and stock['MACD'][i - 1] < stock['MACD Signal'][i - 1]:
                    stock['MACD Trend'][i] = 3.0
                if stock['MACD'][i] > stock['MACD Signal'][i] and stock['MACD'][i - 1] > stock['MACD Signal'][i - 1]:
                    stock['MACD Trend'][i] = 4.0
        else:
            for i in range(len(stock.index)):
                macdList = [stock['MACD'][i - j] for j in range(period, 0, -1)]
                sma = sum(macdList)/period
                stock['MACD Signal'][i] = sma

        return stock

def macd(stock):
    for i in range(len(stock.index)):
        if stock['EMA12'][i] != 0 and stock['EMA26'][i] != 0:
            stock['MACD'][i] = stock['EMA12'][i] - stock['EMA26'][i]
        else:
            stock['MACD'][i] = 0


    return stock

def rsi(stock, period):
    stock['RSI'][0] = 50
    gainsList = []
    lossesList = []
    for i in range(1, period):
        avgGain = sum(stock['pctChange'][i - j + 1] for j in range(i, 0, -1) if stock['pctChange'][i - j + 1] > 0)/i
        avgLoss = -1 * sum(stock['pctChange'][i - j + 1] for j in range(i, 0, -1) if stock['pctChange'][i - j + 1] < 0)/i
        if avgGain == 0:
            avgGain = 1
        if avgLoss == 0:
            avgLoss = 1
        gainsList.append(avgGain)
        lossesList.append(avgLoss)
        value = 100 - (100/(1 + avgGain/avgLoss))
        if value > 100:
            value = 100
        stock['RSI'][i] = value


    for i in range(period, len(stock.index)):
        avgGain = sum(stock['pctChange'][i - j] for j in range(period, 0, -1) if stock['pctChange'][i - j] > 0)/period
        gainsList.append(avgGain)
        avgLoss = -1 * sum(stock['pctChange'][i - j] for j in range(period, 0, -1) if stock['pctChange'][i - j] < 0)/period
        lossesList.append(avgLoss)

        value = 100 - (100/(1 + (gainsList[i - 1] * 13 + stock['pctChange'][i])/(lossesList[i - 1] * 13 + stock['pctChange'][i])))

        if value > 100:
            value = 100
        stock['RSI'][i] = value

        if stock['RSI'][i] < 50.0 and stock['RSI'][i - 1] < 50.0:
            stock['RSI Trend'][i] = 1.0
        if stock['RSI'][i] < 50.0 and stock['RSI'][i - 1] >= 50.0:
            stock['RSI Trend'][i] = 2.0
        if stock['RSI'][i] >= 50.0 and stock['RSI'][i - 1] < 50.0:
            stock['RSI Trend'][i] = 3.0
        if stock['RSI'][i] >= 50.0 and stock['RSI'][i - 1] >= 50.0:
            stock['RSI Trend'][i] = 4.0


    return stock

def stochastic(stock, period):
    lows = []
    highs = []
    high = 0.0
    low = 0.0
    k = 0.0
    kList = [50.0]

    for i in range(1, period):
        lows = [stock['Low'][i - j] for j in range(i, 0, -1)]
        highs = [stock['High'][i - j] for j in range(i, 0, -1)]
        low = min(lows)
        high = max(highs)
        k = 100 * (stock['Adj Close'][i - 1] - low)/(high - low)
        if k > 100:
            k = 100
        kList.append(k)
        stock['Stochastic'][i] = k

    for i in range(period, len(stock.index)):
        lows = [stock['Low'][i - j] for j in range(period, 0, -1)]
        highs = [stock['High'][i - j] for j in range(period, 0, -1)]
        low = min(lows)
        high = max(highs)

        k = 100 * (stock['Adj Close'][i - 1] - low)/(high - low)
        if k > 100:
            k = 100
        kList.append(k)
        stock['Stochastic'][i] = k

    for i in range(period, len(stock.index)):
        stock['Stochastic SMA'][i] = sum(kList[i - j] for j in range(3))/3

    return stock

def convertRSI(percentage):
    value = int(percentage * 0.05)
    if value == 5:
        value = 4
    return value

def convertMACDSignal(difference):
    if difference >= 0:
        return 1
    else:
        return 0
    # table['P(R = )']

# Converts continuous price data to discrete labels and builds out emission-to-emission
# probability matrix
def convert(percentage, probabilities):
    if isinstance(percentage, float):
        if percentage > 0.0010:
            probabilities[0] += 1
            return ["Up", probabilities, 0]
        if percentage < -0.0010:
            probabilities[2] += 1
            return ["Down", probabilities, 2]
        else:
            probabilities[2] += 1
            return ["Neutral", probabilities, 1]

def stockHistory(symbol, startDate, endDate, code):
    stock = pdr.get_data_yahoo(symbol, start=startDate, end=endDate)

    # Assign `Adj Close` to `daily_close`
    daily_close = stock[['Adj Close']]

    # Daily returns
    daily_pct_change = daily_close.pct_change()

    # Replace NA values with 0
    daily_pct_change.fillna(0, inplace=True)

    stock['pctChange'] = stock[['Adj Close']].pct_change()

    # stock['logDelta'] = np.log(daily_close.pct_change()+1)

    stock.fillna(0, inplace=True)

    stock['delta'] = "Test"
    stock['EMA12'] = 0.0
    stock['EMA26'] = 0.0
    stock['MACD'] = 0.0
    stock['MACD Signal'] = 0.0
    stock['MACD Trend'] = 0.0
    stock['RSI'] = 0.0
    stock['RSI Trend'] = 0.0
    stock['Stochastic'] = 0.0
    stock['Stochastic SMA'] = 0.0

    probabilities = [0, 0, 0]
    transitions = [[0,0,0],[0,0,0],[0,0,0]]
    transitionsDict = {}
    for key in [('Up', 'Up'), ('Up', 'Neutral'), ('Up', 'Down')]:
        transitionsDict[key] = 0
    for key in [('Neutral', 'Up'), ('Neutral', 'Neutral'), ('Neutral', 'Down')]:
        transitionsDict[key] = 0
    for key in [('Down', 'Up'), ('Down', 'Neutral'), ('Down', 'Down')]:
        transitionsDict[key] = 0

    sums = 0
    classes = []

    for i in range(0, len(stock.index)):
        # test.append(stock.iloc[i]['pctChange'])
        result = convert(stock.iloc[i]['pctChange'], probabilities)
        stock['delta'][i] = result[0]
        classes.append(result[2])
        if i > 0:
            transitionsDict[(result[0], stock['delta'][i - 1])] += 1
            if stock['delta'][i - 1] is "Up":
                transitions[result[2]][0] += 1
                sums += 1
            elif stock['delta'][i - 1] is "Neutral":
                transitions[result[2]][1] += 1
                sums += 1
            elif stock['delta'][i - 1] is "Down":
                transitions[result[2]][2] += 1
                sums += 1

        probabilities = result[1]

    probabilitiesPct = []
    summed = probabilities[0] + probabilities[1] + probabilities[2]
    probabilitiesPct = [probabilities[0]/summed, probabilities[1]/summed, probabilities[2]/summed]

    print(transitions)
    stock=ema(stock, 12, 'EMA')
    stock=ema(stock, 26, 'EMA')
    stock=macd(stock)
    stock=ema(stock, 9, 'MACD')
    stock=rsi(stock, 14)
    stock=stochastic(stock, 14)

    # print(stock)
    stock.to_csv(symbol + code + ".csv")
    return (stock, classes, transitions)


# """ Naives Bayes Classification """

def naivebayes(ticker):

    stock, classes, transitions = stockHistory(ticker, "2008-01-01", "2018-12-31", "")

    # drop high, low, close, and EMA columns since not useful for NB
    optstock = stock.drop(['High','Low','Close','EMA12','EMA26'], axis=1)
    # optstock = optstock.drop(optstock.index[0:30])

    # Delete last row since can have NaN values
    optstock=optstock.drop(optstock.index[-1])
    classes = classes[:-1]

    # get 80% index to divide data into training and testing sets for fitting
    cutat = int(len(optstock.index) / 10) * 8
    trainingX=optstock.drop(optstock.index[cutat:])
    trainingY=optstock.drop(optstock.index[cutat:])
    trainingY=trainingY.drop(['delta'], axis=1)
    testX=optstock.drop(optstock.index[0:cutat])
    testX = testX.drop(['delta'], axis=1)

    trainingclasses = classes[:cutat]
    testclasses = classes[cutat:]

    dicti = {'R = 0': 0, 'R = 1': 1, 'R = 2': 2, 'R = 3': 3,'R = 4': 4, 'S = 0': 5, 'S = 1': 6, 'S = 2': 7, 'S = 3': 8, 'S = 4': 9, 'M = 0': 10, 'M = 1': 11}
    nrated = []

    # count how many times each delta is present in the training set
    for i in range(3):
        nrated.append(trainingclasses.count(i))

    # Fill counts
    counts = [[0] * len(dicti) for i in range(3)]
    deltatostr = ["Up", "Neutral", "Down"]
    for delta in range(3):
        for i in range(len(trainingX.index)):
            # print(trainingX["delta"][i])
            if trainingX["delta"][i] == deltatostr[delta]:
                if trainingX["RSI"][i] < 20.0:
                    counts[delta][0] += 1
                elif trainingX["RSI"][i] < 40.0:
                    counts[delta][1] += 1
                elif trainingX["RSI"][i] < 60.0:
                    counts[delta][2] += 1
                elif trainingX["RSI"][i] < 80.0:
                    counts[delta][3] += 1
                else:
                    counts[delta][4] += 1

                if trainingX["Stochastic"][i] < 20.0:
                    counts[delta][5] += 1
                elif trainingX["Stochastic"][i] < 40.0:
                    counts[delta][6] += 1
                elif trainingX["Stochastic"][i] < 60.0:
                    counts[delta][7] += 1
                elif trainingX["Stochastic"][i] < 80.0:
                    counts[delta][8] += 1
                else:
                    counts[delta][9] += 1

                if trainingX["MACD"][i] > trainingX["MACD Signal"][i]:
                    counts[delta][10] += 1
                else:
                    counts[delta][11] += 1

    # print("counts")
    # print(counts)
    # print("nrated")
    # print(nrated)

    """ Fitting """
    alpha = 1
    F = [[0] * len(dicti) for i in range(3)]
    for delta in range(3):
        for indicinterv in dicti.values():
            p = (float(alpha)+counts[delta][indicinterv]) / (sum(counts[delta]) + (float(alpha) * len(counts[delta])))
            if p == 0:
                F[delta][indicinterv] = 0
            else:
                F[delta][indicinterv] = -1 * log(p)

    # print("fittt")
    # print(F)

    deltas = []
    accu = 0
    for i in range(len(testX.index)):
        # List of -log(probabilities) of every delta
        l = [0, 0, 0]
        minn = float("inf")
        delta = 10
        r, s, m = 0, 0, 0

        if testX["RSI"][i] < 20.0:
            r = 0
        elif testX["RSI"][i] < 40.0:
            r = 1
        elif testX["RSI"][i] < 60.0:
            r = 2
        elif testX["RSI"][i] < 80.0:
            r = 3
        else:
            r = 4
        if testX["RSI"][i] < 20.0:
            s = 5
        elif testX["RSI"][i] < 40.0:
            s = 6
        elif testX["RSI"][i] < 60.0:
            s = 7
        elif testX["RSI"][i] < 80.0:
            s = 8
        else:
            s = 9
        if testX["MACD"][i] > testX["MACD Signal"][i]:
            m = 10
        else:
            m = 11

        for j in range(3):
            # Calculate the probability through -log
            l[j] = F[j][r] + F[j][s] + F[j][m]
            # Get lowest l (i.e delta with highest prob)
            if l[j] < minn:
                delta = j
                minn = l[j]
        deltas.append(delta)
        # Add an accuracy point if we correctly guessed the delta
        if testclasses[i] == delta:
            accu += 1

        NB = GaussianNB()
        NB.fit(trainingY, trainingclasses)
        predictedclasses = NB.predict(testX)

    implemaccu = float(accu)/float(len(deltas))
    print(ticker)
    if (implemaccu < metrics.accuracy_score(testclasses, predictedclasses)):
        predictions = predictedclasses
        print (metrics.accuracy_score(testclasses, predictedclasses))
    else:
        predictions = deltas
        print(implemaccu)

""" Testing """
naivebayes("NFLX") #48
naivebayes("GOOG") #46
naivebayes("BABA") #38.4
naivebayes("FB") #43
naivebayes("FDX") #46
naivebayes("GE")
naivebayes("GM")
naivebayes("SPY")
naivebayes("QQQ")
naivebayes("BABA")