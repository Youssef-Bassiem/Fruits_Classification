import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import PreProcessing
import tensorflow as tf
from sklearn.metrics import accuracy_score


class MultiNeuralNetwork:
    def __init__(self, classes_count: int, layers_neurons: list, bias_flag: bool, eta: float, threshold: float,
                 activation_func: object):
        self.classes_count = classes_count
        self.layers_neurons = layers_neurons
        self.bias_flag = bias_flag
        self.eta = eta
        self.threshold = threshold
        self.activation_func = activation_func

    def preprocess_data(self, x, y):
        x = x.values.tolist()
        y = y.values.tolist()
        tmp_y = []
        for i in y:
            t = i[0]
            v = np.zeros(shape=(1, self.classes_count))
            v[0, int(t)] = 1
            tmp_y.append(v)
        return x, tmp_y

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def fill_weights(self, inputs: int):
        weights = []
        for layer in range(0, len(self.layers_neurons)):
            m = None
            if layer == 0:
                m = np.random.rand(inputs + int(self.bias_flag), self.layers_neurons[layer])
            else:
                m = np.random.rand(self.layers_neurons[layer - 1] + int(self.bias_flag), self.layers_neurons[layer])
            weights.append(m)
        # output layer weights
        weights.append(np.random.rand(self.layers_neurons[len(self.layers_neurons) - 1] + int(self.bias_flag),
                                      self.classes_count))
        return weights

    def forward_propagation(self, input, layers_weight):
        layers_net: list = []
        for weights in layers_weight:
            if self.bias_flag:
                input = np.insert(input, 0, [[1]], axis=1)
            net = np.dot(input, weights)
            net = self.activation_func(net)
            layers_net.append(net)
            input = net
        return layers_net

    def back_propagation(self, net: list, layers_weight: list, y):
        sigmas: list = []
        output_sigma = net[-1] * (1 - net[-1]) * (y - net[-1])
        sigmas.append(output_sigma)
        for i in range(len(layers_weight) - 2, -1, -1):
            tmp_matrix = np.transpose(layers_weight[i + 1])
            if self.bias_flag:
                tmp_matrix = tmp_matrix[:, 1:]
            sigma = np.dot(sigmas[0], tmp_matrix) * net[i] * (1 - net[i])
            sigmas.insert(0, sigma)
        return sigmas

    def update_weights(self, x, net, layers_weight, sigmas):
        new_layers_weight = []
        for i in range(0, len(layers_weight)):
            input = x
            if i > 0:
                input = net[i - 1]
            if self.bias_flag:
                input = np.insert(input, 0, [[1]], axis=1)
            input = np.transpose(input)
            tmp = self.eta * sigmas[i] * input
            new_weights = layers_weight[i] + tmp
            new_layers_weight.append(new_weights)
        return new_layers_weight

    def train(self, x_train, y_train):
        layers_weight: list = self.fill_weights(x_train.columns.size)
        x_train, y_train = self.preprocess_data(x_train, y_train)
        # x_train = x_train.values.tolist()
        # y_train = y_train.values.tolist()
        MSE = 0
        while True:
            for i in range(0, len(x_train)):
                x_row = np.array(x_train[i])
                x_row = x_row.reshape(1, x_row.shape[0])
                net = self.forward_propagation(x_row, layers_weight)
                sigmas = self.back_propagation(net, layers_weight, y_train[i])
                layers_weight = self.update_weights(x_row, net, layers_weight, sigmas)
                net = self.forward_propagation(x_row, layers_weight)
                MSE += ((y_train[i] - net[-1]) ** 2)
                print(y_train[i])
                print(net[-1])
                print('=======================')
            MSE = MSE / len(x_train)
            print('####################')
            if MSE <= self.threshold:
                break
        return layers_weight, MSE


    # x = pd.DataFrame({
#     'x1': [0, 1, 1, 0],
#     'x2': [0, 1, 0, 1]
# })
# y = pd.DataFrame({'y': [0, 0, 1, 1]})

model = MultiNeuralNetwork(3,
                           [4,4,4,4,4],
                           True,
                           2,
                           0.01,
                           MultiNeuralNetwork.sigmoid)
x_train, y_train, x_test, y_test = PreProcessing.main()
layers_weight, MSE = model.train(x_train, y_train)
sample = np.array([45504, 793.417, 295.46983055204, 196.31182248937, 0.90835672452768])
sample = sample.reshape(1,5)
# print(sample.shape)
net = model.forward_propagation(sample, layers_weight)
print(net[-1])
# print(layers_weight)
# print(MSE)

# plt.scatter([0, 0, 1, 1], [0, 1, 0, 1])
# plt.grid(True)
#
# axis11 = np.arange(0, 1.1, 0.1)
# axis12 = (axis11 * layers_weight[0][1, 0] + layers_weight[0][0, 0]) / (-1 * layers_weight[0][2, 0])
# plt.plot(axis11, axis12)
#
# axis21 = np.arange(0, 1.1, 0.1)
# axis22 = (axis21 * layers_weight[0][1, 1] + layers_weight[0][0, 1]) / (-1 * layers_weight[0][2, 1])
# plt.plot(axis21, axis22)

# axis31 = np.arange(0, 1.1, 0.1)
# axis32 = (axis31 * layers_weight[1][1, 0] + layers_weight[1][0, 0]) / (-1 * layers_weight[1][2, 0])
# plt.plot(axis31, axis32)

# plt.show()
