# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 22:30:06 2023

@author: fat7i nasser
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

from sklearn import preprocessing

samples = 50
trainSamples = 30
testSamples = samples - trainSamples

C1 = 2
C2 = 0

flagOfBias = True

features = ['Class', 'Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'roundnes']


def split_data(start, all_data):
    return (all_data.loc[start: start + samples - 1, features].
            sample(frac=1, random_state=42)).reset_index(drop=True)

l=0.01

def signum(value):
    return 1 if value >= 0 else -1


def fillnull(data):
    for j in range(1, len(features)):
        for i in range(0, data.shape[0]):
            x = data.loc[i, features[j]]
            x=data[features[j]]
            if  np.isnan(data.loc[i, features[j]]):
                data.loc[i, features[j]]=data[features[j]].median()

    return data

def getminmax(data,xa):
    minmax=[]
    for j in range(1, len(features)):
        for i in range(0, data.shape[0]):
            xa[i]=data.loc[i, features[j]]
        value_min=min(xa)
        value_max=max(xa)
        minmax.append([value_min, value_max])
    return minmax

def moralizes(data,minmax):
    for j in range(0, len(features)-1):
        for i in range(0, data.shape[0]):
            data.loc[i, features[j+1]] = (data.loc[i, features[j+1]] - minmax[j][0]) / (minmax[j][1] - minmax[j][0])

    return data
def stop(w,data,xi):
    errors = 0
    for i in range(0, data.shape[0]):
        for j in range(1, len(features)):
            xi[j] = data.loc[i, features[j]]
        y_hat = np.dot(w.reshape(1, -1), xi.reshape(-1, 1))
        errors = errors + ((data.loc[i, features[0]] - y_hat) ** 2)
    mse = (1 / data.shape[0]) * (errors / 2)
    return mse

def train_model(w,data,xi):
    epochs=3000
    while epochs!=0:
        for i in range(0,data.shape[0]):
            for j in range(1, len(features)):
                xi[j]=data.loc[i,features[j]]
            y_hat=np.dot(xi.reshape(1,-1),w.reshape(-1,1))
            if y_hat[0][0] != data.loc[i,features[0]]:
                error=data.loc[i,features[0]]-y_hat
                w=w.reshape(-1,1)+(l*error[0][0]*xi.reshape(-1,1))
        mse=stop(w,data,xi)
        print(mse)
        if(mse<0.2):
            break
        epochs=epochs-1
    return w

            

def test_model(w, y, test_data, xi):
    incorrect = 0
    correct = 0

    for i in range(trainSamples, test_data.shape[0]):
        for k in range(1, len(features)):
            xi[k] = test_data.loc[i, features[k]]
        y_hat = signum(np.dot(xi.reshape(1, -1), w.reshape(-1, 1)))
        print(y_hat)
        if y_hat != y:
            incorrect += 1
        else:
            correct += 1
    return correct




def main():
    df = pd.read_csv("D:\\fcis2024 7th\\CV-20231016T055155Z-001\CV\\Labs\Lab6//Dry_Bean_Dataset.csv")
    startOfC1 = C1 * trainSamples
    dataOfC1 = split_data(startOfC1, df)
    c1_class = dataOfC1['Class']
    dataOfC1.drop(['Class'], axis=1, inplace=True)
    dataOfC1 = fillnull(dataOfC1)

    xa=np.zeros(dataOfC1.shape[0])
    mima=getminmax(dataOfC1,xa)

    dataOfC1 = moralizes(dataOfC1,mima)




    dataOfC1.insert(0, 'Class', c1_class.tolist())
    #dataOfC1=fillnull(dataOfC1)
    samplesOfC1 = dataOfC1.loc[0: trainSamples - 1, features]

    startOfC2 = C2 * trainSamples
    dataOfC2 = split_data(startOfC2, df)
    c2_class = dataOfC2['Class']
    dataOfC2.drop(['Class'], axis=1, inplace=True)

    dataOfC2 = fillnull(dataOfC2)

    xa = np.zeros(dataOfC2.shape[0])
    mima2 = getminmax(dataOfC2, xa)


    dataOfC2 = moralizes(dataOfC2,mima2)

    dataOfC2.insert(0, 'Class', c2_class.tolist())

    samplesOfC2 = dataOfC2.loc[0: trainSamples - 1, features]

    samplesOfC1[samplesOfC1.columns[0]] = np.ones(samplesOfC1.shape[0])
    samplesOfC2[samplesOfC2.columns[0]] = np.ones(samplesOfC2.shape[0]) * -1

    data = (pd.concat([samplesOfC1, samplesOfC2]).
            sample(frac=1, random_state=42).reset_index(drop=True))


    xi = np.ones(len(features))
    xi[0] = 1 if flagOfBias else 0

    weights = np.zeros(len(features))
    for i in range(0, len(weights)):
        weights[i] = np.random.rand()
    weights=train_model(weights, data, xi)
    print(weights)
    print("Correct: ", test_model(w=weights, y=samplesOfC1.iloc[0, 0], test_data=dataOfC1, xi=xi), "From", testSamples)
    print("Correct: ", test_model(w=weights, y=samplesOfC2.iloc[0, 0], test_data=dataOfC2, xi=xi), "From", testSamples)
    return dataOfC1, dataOfC2, weights


if __name__ == "__main__":
    main()