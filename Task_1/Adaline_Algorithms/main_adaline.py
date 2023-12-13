import numpy as np
from Task_1.preprocessing import pre

samples = 50
trainSamples = 30
testSamples = samples - trainSamples

C1 = 2
C2 = 1

flagOfBias = False
epochs = 100
L = 0.1
threshold = 0.001
features = ['Class', 'Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'roundnes']


def get_mse(w, data, xi):
    errors = 0
    for i in range(0, data.shape[0]):
        for j in range(1, len(features)):
            xi[j] = data.loc[i, features[j]]
        y_hat = np.dot(w.reshape(1, -1), xi.reshape(-1, 1))
        errors = errors + ((data.loc[i, features[0]] - y_hat) ** 2)
    mse = (1 / 2) * (errors / data.shape[0])
    return mse


def train_model(w, data, xi):
    for _ in range(epochs):
        for j in range(0, data.shape[0]):
            for k in range(1, len(features)):
                xi[k] = data.loc[j, features[k]]
            y_hat = (np.dot(xi.reshape(1, -1), w.reshape(-1, 1))[0][0])
            x = (data.iloc[j, 0])
            if y_hat != x:
                error = (data.iloc[j, 0] - y_hat)
                w += L * error * xi
        mse = get_mse(w, data, xi)
        if mse[0][0] < threshold:
            break
    return w


def main():
    weights = np.zeros(len(features))
    for i in range(0, len(weights)):
        weights[i] = np.random.rand()
    xi = np.ones(len(features))
    xi[0] = 1 if flagOfBias else 0
    dd1, dd2, data = pre(C1, C2, samples, trainSamples, features)
    weights = train_model(weights, data, xi)
    return dd1, dd2, weights


if __name__ == "__main__":
    main()
