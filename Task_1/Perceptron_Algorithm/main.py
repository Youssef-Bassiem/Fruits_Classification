import matplotlib as plt
import pandas as pd
import numpy as np
import math

from sklearn import preprocessing

samples = 50
trainSamples = 30
testSamples = samples - trainSamples

C1 = 0
C2 = 1

flagOfBias = False
epochs = 250
L = 0.00001

features = ['Class', 'Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'roundnes']


def split_data(start, all_data):
    return (all_data.loc[start: start + samples - 1, features].
            sample(frac=1, random_state=42)).reset_index(drop=True)


def signum(value):
    if(value>0):
        return 1
    elif(value<0):
        return -1
    else:
        return 0



def train_model(w, data, xi):
    for _ in range(epochs):
        for j in range(0, data.shape[0]):
            for k in range(1, len(features)):
                xi[k] = data.loc[j, features[k]]



            y_hat = signum(np.dot(xi.reshape(1, -1), w.reshape(-1, 1)))
            if y_hat != data.iloc[j, 0]:
                error = (data.iloc[j, 0] - y_hat)
                w += L * error * xi
    return w


def function_mesh(w, y, test_data, xi):
    incorrect = 0
    correct = 0

    for j in range(trainSamples, test_data.shape[0]):
        for k in range(1, len(features)):
            xi[k] = test_data.loc[j, features[k]]
        y_hat = signum(np.dot(xi.reshape(1, -1), w.reshape(-1, 1)))

        if y_hat != y:
            incorrect += 1
        else:
            correct += 1
    return correct


def normalize(data):
    cols = data.columns
    x = data.values  # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled)
    df.columns = cols
    return df


def main():
    df = pd.read_excel("../Dry_Bean_Dataset.xlsx")
    startOfC1 = C1 * samples
    dataOfC1 = split_data(startOfC1, df)
    c1_class = dataOfC1['Class']
    dataOfC1.drop(['Class'], axis=1, inplace=True)
    # dataOfC1=normalize(dataOfC1)
    dataOfC1.insert(0, 'Class', c1_class.tolist())
    dataOfC1['Class'] =1


    samplesOfC1 = dataOfC1.loc[0: trainSamples - 1, features]

    startOfC2 = C2 * samples
    dataOfC2 = split_data(startOfC2, df)
    c2_class = dataOfC2['Class']
    dataOfC2.drop(['Class'], axis=1, inplace=True)
    # dataOfC2 = normalize(dataOfC2)
    dataOfC2.insert(0, 'Class', c2_class.tolist())
    dataOfC2['Class']=(-1)

    samplesOfC2 = dataOfC2.loc[0: trainSamples - 1, features]

    samplesOfC1[samplesOfC1.columns[0]] = np.ones(samplesOfC1.shape[0])
    samplesOfC2[samplesOfC2.columns[0]] = np.ones(samplesOfC2.shape[0])*-1





    data = (pd.concat([samplesOfC1, samplesOfC2]).
            sample(frac=1, random_state=42).reset_index(drop=True))
    ##########################


    for feature in features:
        data[feature].fillna(value=data[feature].median(), inplace=True)




    ###########################


    xi = np.ones(len(features))
    xi[0] = 1 if flagOfBias else 0


    weights = np.zeros(len(features))
    for i in range(0, len(weights)):
        # weights[i] = np.random.rand()
        weights[i] = 1

    weights = train_model(weights, data, xi)
    # print(weights)
    print("Correct: ", function_mesh(w=weights, y=samplesOfC1.iloc[0, 0], test_data=dataOfC1, xi=xi), "From", testSamples)
    print("Correct: ", function_mesh(w=weights, y=samplesOfC2.iloc[0, 0], test_data=dataOfC2, xi=xi), "From", testSamples)
    # dataOfC1.drop(['Class'], axis=1, inplace=True)
    # dataOfC2.drop(['Class'], axis=1, inplace=True)
    return dataOfC1, dataOfC2, weights


if __name__ == "__main__":
    main()