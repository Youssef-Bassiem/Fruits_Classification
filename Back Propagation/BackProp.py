import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import PreProcessing
# import tensorflow as tf
from sklearn.metrics import accuracy_score
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")


class MultiNeuralNetwork:
    def __init__(self, layers_neurons: list, bias_flag: bool, eta: float, threshold: float,
                 activation_func: object):
        self.layers_neurons = layers_neurons
        self.bias_flag = bias_flag
        self.eta = eta
        self.threshold = threshold
        self.activation_func = activation_func

    def preprocess_input(self, x, y, classes_count):
        x = np.array(x)
        y = np.array(y)
        tmp_y = []
        for i in y:
            t = i[0]
            v = np.zeros(shape=(1, classes_count))
            v[0, int(t)] = 1
            tmp_y.append(v)
        return x, tmp_y

    def preprocess_output(self, y):
        for i in range(y.shape[1]):
            if y[0, i] == y.max():
                return i


    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def tanh_derivative(x):
        return 1 - x ** 2

    @staticmethod
    def softmax(x):
        return np.exp(x) / np.sum(np.exp(x))

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    def fill_weights(self, inputs: int, classes_count):
        weights = []
        for layer in range(0, len(self.layers_neurons)):
            m = None
            if layer == 0:
                m = np.random.rand(inputs + int(self.bias_flag), self.layers_neurons[layer])
            else:
                m = np.random.rand(self.layers_neurons[layer - 1] + int(self.bias_flag), self.layers_neurons[layer])
            weights.append(m)
        # output layer weights
        weights.append(np.random.rand(self.layers_neurons[-1] + int(self.bias_flag), classes_count))
        return weights

    def forward_propagation(self, input, layers_weight):
        layers_net: list = []
        for weights in layers_weight:
            if self.bias_flag:
                input = np.insert(input, 0, [[1]], axis=1)
            net = np.dot(input, weights)
            if weights is layers_weight[-1]:
                net = self.tanh(net)
            else:
                net = self.tanh(net)
            layers_net.append(net)
            input = net
        return layers_net

    def back_propagation(self, net: list, layers_weight: list, y):
        sigmas: list = []
        output_sigma = self.tanh_derivative(net[-1]) * (y - net[-1])
        sigmas.append(output_sigma)
        for i in range(len(layers_weight) - 2, -1, -1):
            tmp_matrix = np.transpose(layers_weight[i + 1])
            if self.bias_flag:
                tmp_matrix = tmp_matrix[:, 1:]
            sigma = np.dot(sigmas[0], tmp_matrix) * self.tanh_derivative(net[i])
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
            # input = np.transpose(input)
            tmp = self.eta * sigmas[i] * np.transpose(input)
            new_weights = layers_weight[i] + tmp
            new_layers_weight.append(new_weights)
        return new_layers_weight

    # def testing_accuracy(self, activation_function, inputs, outputs):
    #     inputs, outputs = self.preprocess_input(inputs, outputs, int(outputs.max()+1))
    #     predictions = []
    #     predictions = self.accuracy(inputs, activation_function)
    #     accuracy = accuracy_score(outputs, predictions)
    #     return accuracy
    #
    # def accuracy(self, inputs, activation_function):
    #
    #     pred = []
    #     for i in range(len(inputs)):
    #         layer_output = self.forward_propagation(inputs[i], activation_function)
    #         predicted_class = np.argmax(layer_output[-1])
    #         pred.append(predicted_class)
    #     return pred

    def accuracy(self, x, y, layers_weights):
        y = np.array(y)
        x, tmp_y = self.preprocess_input(x, y, int(y_train.max()+1))
        true_predictions = 0
        for i in range(x.shape[0]):
            x_sample = x[i].reshape(1, x.shape[1])
            net = self.forward_propagation(x_sample, layers_weights)
            y_predict = self.preprocess_output(net[-1])
            if y_predict == y[i, 0]:
                true_predictions += 1
        return (true_predictions / y.shape[0]) * 100



    def train(self, x_train, y_train):
        layers_weight: list = self.fill_weights(x_train.columns.size, int(y_train.max())+1)
        x_train, y_train = self.preprocess_input(x_train, y_train, int(y_train.max()) + 1)
        MSE = 0
        for e in range(1000):
            for i in range(0, x_train.shape[0]):
                x_row = x_train[i].reshape(1, x_train.shape[1])
                net = self.forward_propagation(x_row, layers_weight)
                error = np.abs(y_train[i] - net[-1]).mean()
                if error >= 0.1:
                    sigmas = self.back_propagation(net, layers_weight, y_train[i])
                    layers_weight = self.update_weights(x_row, net, layers_weight, sigmas)
                MSE += (np.abs(y_train[i] - net[-1])).mean()
            MSE = MSE / x_train.shape[0]
            # print(MSE)
            if MSE <= self.threshold:
                break
        return layers_weight, MSE


x = pd.DataFrame({
    'x1': [0, 1, 1, 0],
    'x2': [0, 1, 0, 1]
})
y = pd.DataFrame({'y': [0, 0, 1, 1]})

model = MultiNeuralNetwork([5],
                           True,
                           0.01,
                           0.01,
                           MultiNeuralNetwork.sigmoid)
x_train, y_train, x_test, y_test = PreProcessing.main()
layers_weights, MSE = model.train(x_train, y_train)
print('train accuracy : ', model.accuracy(x_train, y_train, layers_weights))
print('test accuracy : ', model.accuracy(x_test, y_test, layers_weights))
# print('-----------')
# print(layers_weights)
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
#
# print("***********************")
#
# x_test = np.array(x_test)
# y_test = np.array(y_test)
# for i in range(x_test.shape[0]):
#     sample = x_test[i].reshape(1, 5)
#     print(x_test[i])
#     print(y_test[i])
#     print(model.forward_propagation(sample, layers_weight)[-1])
#     print('-----------------------------')

# print("***********************")
