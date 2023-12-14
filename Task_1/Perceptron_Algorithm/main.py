import pandas as pd
import numpy as np
from Task_1.preprocessing import pre

from sklearn import preprocessing

samples = 50
trainSamples = 30
testSamples = samples - trainSamples

C1 = 0
C2 = 2

flagOfBias = False
epochs = 550
L = 0.01

features = ['Class', 'Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'roundnes']


def split_data(start, all_data):
    return (all_data.loc[start: start + samples - 1, features].
            sample(frac=1, random_state=42)).reset_index(drop=True)


def signum(value):
    if value > 0:
        return 1
    elif value < 0:
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
    # print(test_data)
    for j in range(0, test_data.shape[0]):
        for k in range(1, len(features)):
            # print(test_data)
            # print(test_data.loc[30, features[k]])
            xi[k] = test_data.loc[j, features[k]]
        y_hat = signum(np.dot(xi.reshape(1, -1), w.reshape(-1, 1)))
        # print(y)
        # print(y_hat)
        if y_hat != y:
            incorrect += 1
        else:
            correct += 1
    return correct


def normalize(data):
    cols = data.columns
    x = data.values  # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled)
    df.columns = cols
    return df

def main():
    weights = np.zeros(len(features))
    for i in range(0, len(weights)):
        weights[i] = np.random.rand()
    xi = np.ones(len(features))
    xi[0] = 1 if flagOfBias else 0
    dd1, dd2, data, samples_of_c1, samples_of_c2, test_of_c1, test_of_c2 = pre(C1, C2, samples, trainSamples, features)
    # print(samples_of_c1)
    # print(data)
    # print(samples_of_c2)
    weights = train_model(weights, data, xi)
    # print("Correct: ", function_mesh(w=weights, y=test_of_c1.iloc[0, 0], test_data=test_of_c1, xi=xi), "From",
    #       testSamples)
    # print("Correct: ", function_mesh(w=weights, y=test_of_c2.iloc[0, 0], test_data=test_of_c2, xi=xi), "From",
    #       testSamples)
    # print(data)
    print(weights)
    correct1 = function_mesh(w=weights, y=samples_of_c1.iloc[0, 0], test_data=test_of_c1, xi=xi)
    correct2 = function_mesh(w=weights, y=samples_of_c2.iloc[0, 0], test_data=test_of_c2, xi=xi)
    print("Correct:", correct1, "From", testSamples, "Incorrect", testSamples - correct1)
    print("Correct:", correct2, "From", testSamples, "Incorrect", testSamples - correct2)
    print("accuracy:", (correct1+correct2)/(testSamples*2) * 100)
    return dd1, dd2, weights


if __name__ == "__main__":
    main()
