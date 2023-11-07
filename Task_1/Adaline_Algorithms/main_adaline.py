import pandas as pd
import numpy as np
import math

from sklearn import preprocessing

samples = 50
trainSamples = 30
testSamples = samples - trainSamples

C1 = 2
C2 = 0
threshold = 0.6
flagOfBias = True
epochs = 150
L = 0.0001

features = ['Class', 'Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'roundnes']


def split_data(start, all_data):
    return (all_data.loc[start: start + samples - 1, features].
            sample(frac=1, random_state=42)).reset_index(drop=True)


def signum(value):
    return 1 if value >= 0 else -1


def train_model(w, data, xi):
    while True:
        total_error = 0
        for j in range(0, data.shape[0]):
            for k in range(1, len(features)):
                xi[k] = 0 if (data.loc[j, features[k]].mean()) else data.loc[j, features[k]]
            y_hat = np.dot(xi.reshape(1, -1), w.reshape(-1, 1))
            if y_hat[0] != data.iloc[j, 0]:
                error = (data.iloc[j, 0] - y_hat[0])
                w += L * error * xi
                total_error += (data.iloc[j, 0] - y_hat[0]) ** 2
        mse = (1 / 2) * (total_error / data.shape[0])
        if mse <= threshold:
            break
    return w

def test_model(w, y, test_data, xi):
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


def normalize(data):
    minVal = data.min().iloc[1:data.shape[0]].min()
    maxVal = data.max().iloc[1:data.shape[0]].max()
    # for f in range(1 ,len(features)):
    #     for i in range(0, len(data[features[f]])):
    #         data[features[f]][i] = (data[features[f]][i] - data[features[f]].min()) / (data[features[f]].max() - data[features[f]].min())
    print(data[['Area','Perimeter']])
    cols = data.columns
    x = data.values  # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled)
    df.columns = cols
    return df
    # data.iloc[0:data.shape[0], 1:] = (data.iloc[0:data.shape[0], 1:] - minVal) / (maxVal - minVal)


def main():
    df = pd.read_excel("../Dry_Bean_Dataset.xlsx")
    # df = pd.read_excel('Task_1/Perceptron_Algorithm/Dry_Bean_Dataset.xlsx')
    startOfC1 = C1 * trainSamples
    dataOfC1 = split_data(startOfC1, df)
    c1_class = dataOfC1['Class']
    dataOfC1.drop(['Class'], axis = 1, inplace = True)
    # dataOfC1 = normalize(dataOfC1)
    dataOfC1.insert(0, 'Class', c1_class.tolist())

    samplesOfC1 = dataOfC1.loc[0: trainSamples - 1, features]

    startOfC2 = C2 * trainSamples
    dataOfC2 = split_data(startOfC2, df)
    c2_class = dataOfC2['Class']
    dataOfC2.drop(['Class'], axis = 1, inplace = True)
    # dataOfC2 = normalize(dataOfC2)
    dataOfC2.insert(0, 'Class', c2_class.tolist())

    # plt.scatter(dataOfC1['Area'], dataOfC1['Perimeter'])
    # plt.scatter(dataOfC2['Area'], dataOfC2['Perimeter'])
    # plt.show()


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
    weights = train_model(weights, data, xi)
    print(weights)
    print("Correct: ", test_model(w=weights, y=samplesOfC1.iloc[0, 0], test_data=dataOfC1, xi=xi), "From", testSamples)
    print("Correct: ", test_model(w=weights, y=samplesOfC2.iloc[0, 0], test_data=dataOfC2, xi=xi), "From", testSamples)
    return dataOfC1, dataOfC2, weights



if __name__ == "__main__":
    main()