import pandas as pd
import numpy as np
import math

samples = 50
trainSamples = 30
testSamples = samples - trainSamples

C1 = 2
C2 = 0

flagOfBias = True
epochs = 150
L = 0.0001

features = ['Class', 'Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'roundnes']


def split_data(start, all_data):
    return (all_data.loc[start: start + samples - 1, features].
            sample(frac=1, random_state=42)).reset_index(drop=True)


def signum(value):
    return 1 if value >= 0 else -1

threshold=0.01
def train_model(w):
    
    while(True):
        for j in range(0, data.shape[0]):
            for k in range(1, len(features)):
                xi[k] = 0 if math.isnan(data.loc[j, features[k]]) else data.loc[j, features[k]]
            y_hat = np.dot(xi.reshape(1, -1), w.reshape(-1, 1))
            if y_hat != data.iloc[j, 0]:
                error = (data.iloc[j, 0] - y_hat)
                w += L * error * xi
        for i in range(0, data.shape[0]):
            for c in range(1, len(features)):
                xi[c] = 0 if math.isnan(data.loc[i, features[k]]) else data.loc[i, features[k]]
            y_hat = np.dot(xi.reshape(1, -1), w.reshape(-1, 1))
            if y_hat != data.iloc[i, 0]:
                error += (data.iloc[i, 0] - y_hat)**2
        mses=(1/2)*(error/data.shape[0])
        if(mses<=threshold):
            break
                
    return w





        
def test_model(w, y, test_data):
    incorrect = 0
    correct = 0

    for j in range(trainSamples, test_data.shape[0]):
        for k in range(1, len(features)):
            xi[k] = 0 if math.isnan(xi[k]) else test_data.loc[j, features[k]]
        y_hat = signum(np.dot(xi.reshape(1, -1), w.reshape(-1, 1)))

        if y_hat != y:
            incorrect += 1
        else:
            correct += 1
    return correct


if __name__ == '__main__':
    df = pd.read_excel("Dry_Bean_Dataset.xlsx")

    startOfC1 = C1 * trainSamples
    dataOfC1 = split_data(startOfC1, df)
    samplesOfC1 = dataOfC1.loc[0: trainSamples - 1, features]

    startOfC2 = C2 * trainSamples
    dataOfC2 = split_data(startOfC2, df)
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
    weights = train_model(weights)
    print(weights)
    print("Correct: ", test_model(w=weights, y=samplesOfC1.iloc[0, 0], test_data=dataOfC1), "From", testSamples)
    print("Correct: ", test_model(w=weights, y=samplesOfC2.iloc[0, 0], test_data=dataOfC2), "From", testSamples)