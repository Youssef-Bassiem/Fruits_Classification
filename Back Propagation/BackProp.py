import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import PreProcessing
# import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")


class MultiNeuralNetwork:
    def __init__(self, layers_neurons: list, bias_flag: bool, eta: float, activ_func, activ_func_deriv):
        self.layers_neurons = layers_neurons
        self.bias_flag = bias_flag
        self.eta = eta
        self.activ_func = activ_func
        self.activ_func_deriv = activ_func_deriv

    def preprocess_input(self, x, y):
        x = np.array(x)
        y = np.array(y)
        classes_count = int(y.max() + 1)
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
                net = self.activ_func(net)
            else:
                net = self.activ_func(net)
            layers_net.append(net)
            input = net
        return layers_net

    def back_propagation(self, net: list, layers_weight: list, y):
        sigmas: list = []
        output_sigma = self.activ_func_deriv(net[-1]) * (y - net[-1])
        sigmas.append(output_sigma)
        for i in range(len(layers_weight) - 2, -1, -1):
            tmp_matrix = np.transpose(layers_weight[i + 1])
            if self.bias_flag:
                tmp_matrix = tmp_matrix[:, 1:]
            sigma = np.dot(sigmas[0], tmp_matrix) * self.activ_func_deriv(net[i])
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

    def test(self, data, layers_weights):
        data = np.array(data)
        predictions = []
        for i in range(data.shape[0]):
            sample = data[i].reshape(1, data.shape[1])
            net = self.forward_propagation(sample, layers_weights)
            predictions.append(self.preprocess_output(net[-1]))
        return predictions

    def accuracy(self, x, y, layers_weights):
        y = np.array(y)
        x = np.array(x)
        true_predictions = 0
        predictions = self.test(x, layers_weights)
        for i in range(len(predictions)):
            if predictions[i] == y[i, 0]:
                true_predictions += 1
        return (true_predictions / y.shape[0]) * 100

    def confusion_matrix(self, y_test, y_pred):
        t11 = 0
        t12 = 0
        t13 = 0
        t21 = 0
        t22 = 0
        t23 = 0
        t31 = 0
        t32 = 0
        t33 = 0

        for a, p in zip(y_test['Class'], y_pred):
            a = int(a)
            p = int(p)
            if a == 0 and p == 0:
                t11 += 1
            elif a == 0 and p == 1:
                t12 += 1
            elif a == 0 and p == 2:
                t13 += 1
            elif a == 1 and p == 0:
                t21 += 1
            elif a == 1 and p == 1:
                t22 += 1
            elif a == 1 and p == 2:
                t23 += 1
            elif a == 2 and p == 0:
                t31 += 1
            elif a == 2 and p == 1:
                t32 += 1
            elif a == 2 and p == 2:
                t33 += 1
            # Create the confusion matrix
        matrix = [
            [t11, t12, t13],
            [t21, t22, t23],
            [t31, t32, t33]
        ]

        # Display the confusion matrix
        print("Confusion Matrix: ")
        # print(matrix)
        for row in matrix:
            for value in row:
                print(f"{value:2}", end=" ")  # Adjust the formatting as needed
            print()

    def train(self, x_train, y_train):
        layers_weight: list = self.fill_weights(x_train.columns.size, int(y_train.max()) + 1)
        x_train, y_train = self.preprocess_input(x_train, y_train)
        for epoch in range(500):
            for i in range(0, x_train.shape[0]):
                x_row = x_train[i].reshape(1, x_train.shape[1])
                net = self.forward_propagation(x_row, layers_weight)
                # error = np.abs(y_train[i] - net[-1]).mean()
                sigmas = self.back_propagation(net, layers_weight, y_train[i])
                layers_weight = self.update_weights(x_row, net, layers_weight, sigmas)
        return layers_weight


x = pd.DataFrame({
    'x1': [0, 1, 1, 0],
    'x2': [0, 1, 0, 1]
})
y = pd.DataFrame({'y': [0, 0, 1, 1]})

model = MultiNeuralNetwork([3],
                           True,
                           0.1,
                           MultiNeuralNetwork.sigmoid,
                           MultiNeuralNetwork.sigmoid_derivative
                           )
x_train, y_train, x_test, y_test = PreProcessing.main()
layers_weights = model.train(x_train, y_train)
y_pred = model.test(x_test, layers_weights)

print('train accuracy : ', model.accuracy(x_train, y_train, layers_weights))
print('test accuracy : ', model.accuracy(x_test, y_test, layers_weights))
print("*****************************************************************************")
print("-------confusion matrix-------")
print(confusion_matrix(y_test, y_pred))
print("++++++++++++++++++++++++++++++++")
model.confusion_matrix(y_test, y_pred)
print("*****************************************************************************")


