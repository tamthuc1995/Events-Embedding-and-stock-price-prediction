#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 16:57:11 2017

@author: red-sky
"""
import sys
import numpy as np
import pickle
import pandas as pd


def main(VectorsPath, EventPath, StockPricePath, days):

    with open(a, "rb") as H:
        Vec = pickle.load(H)
        Vectors = np.array([list(b[0]) for a, b in Vec.values()])
#    Vectors = np.load(VectorsPath)
    with open(EventPath, "r") as H:
        F = np.array([a.split("\t")[0:4] for a in H.read().splitlines()])

    D = {}
    for date, vec in zip(F[:, 0], Vectors):
        if date[:10] in D:
            D[date[:10]].append(vec)
        else:
            D[date[:10]] = [vec]

    D2 = {}
    for date in sorted(D.keys()):
        D2[date] = np.mean(D[date], 0)

    Dates = np.array(sorted(D2.keys()))
    SampleIndex = [list(range(i-days, i)) for i in range(5, len(Dates))]
    DataX = []
    DateX = []
    for listIndex in SampleIndex:
        DataX.append([D2[date] for date in Dates[listIndex]])
        DateX.append(Dates[listIndex[-1]])

    Df = pd.read_csv(StockPricePath)
    LabelY = []
    DataX_yesData = []
    for i, date in enumerate(DateX):
        retu = list(Df.loc[Df["Date"] == date]["ReturnOpen"])
        print(retu)
        if len(retu) > 0:
            retu = float(retu[0])*100
            if retu > 0:
                LabelY.append([1, 0])
            if retu < -0:
                LabelY.append([0, 1])
            if retu <= 0 and retu >= -0:
                LabelY.append([0, 1])
            DataX_yesData.append(list(DataX[i]))
            print(date)
#        else:

    dataX = np.array(DataX_yesData)
    dataY = np.array(LabelY)
    print("DataX:", dataX.shape)
    print("DataY:", dataY.shape, np.sum(dataY, 0) / np.sum(dataY))
    return (dataX, dataY)

if __name__ == "__main__":
    VectorsPath = sys.argv[1]
    EventPath = sys.argv[2]
    StockPricePath = sys.argv[3]
    days = int(sys.argv[5])
    DataX, LabelY = main(VectorsPath, EventPath, StockPricePath, days)
    DataPath = sys.argv[4]
    np.save(arr=DataX, file=DataPath+"/DailyVector" + sys.argv[5] + ".npy")
    np.save(arr=LabelY, file=DataPath+"/DailyReturn" + sys.argv[5] + ".npy")
