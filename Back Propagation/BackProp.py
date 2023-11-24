import numpy as np
import matplotlib.pyplot as plt



inputs = 2
layers_count = 1
neurons_num = [2]
output_neurons = 1
bias_flag = True
eta = 0.1
threshold = 0.01
x = [
    np.array([[0, 0]]),
    np.array([[1, 1]]),
    np.array([[1, 0]]),
    np.array([[0, 1]])
]
y = np.array([[0, 0, 1, 1]] )


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def activation_function(v):  # v -> vector of net output
    return sigmoid(v)


def fill_weights(neurons_num: list, layers_count: int, inputs: int, bias_flag: bool, output_neurons: int):
    weights = []
    for layer in range(0, layers_count):
        m = None
        if layer == 0:
            m = np.random.rand(inputs + int(bias_flag), neurons_num[layer])
        else:
            m = np.random.rand(neurons_num[layer - 1] + int(bias_flag), neurons_num[layer])
        weights.append(m)
    # output layer weights
    weights.append(np.random.rand(neurons_num[layers_count - 1] + int(bias_flag), output_neurons))
    return weights


def forward_propagation(input, layers_weight, bias_flag):
    layers_net: list = []
    for weights in layers_weight:
        if bias_flag:
            input = np.insert(input, 0, [[1]], axis=1)
        net = np.dot(input, weights)
        net = activation_function(net)
        layers_net.append(net)
        input = net
    return layers_net


def back_propagation(net: list, layers_weight: list, y, bias_flag):
    sigmas: list = []
    output_sigma = net[-1] * (1 - net[-1]) * (y - net[-1])
    sigmas.append(output_sigma)
    for i in range(len(layers_weight) - 2, -1, -1):
        tmp_matrix = np.transpose(layers_weight[i + 1])
        if bias_flag:
            tmp_matrix = tmp_matrix[:, 1:]
        sigma = np.dot(sigmas[0], tmp_matrix) * net[i] * (1 - net[i])
        sigmas.insert(0, sigma)
    return sigmas


def update_weights(x, net, layers_weight, sigmas):
    new_layers_weight = []
    for i in range(0, len(layers_weight)):
        input = x
        if i > 0:
            input = net[i - 1]
        if bias_flag:
            input = np.insert(input, 0, [[1]], axis=1)
        input = np.transpose(input)
        tmp = eta * sigmas[i] * input
        new_weights = layers_weight[i] + tmp
        new_layers_weight.append(new_weights)
    return new_layers_weight


def train(x, bias_flag):
    layers_weight: list = fill_weights(neurons_num, layers_count, inputs, bias_flag, output_neurons)
    # layers_weight = [np.array([[0.4, -0.1],
    #                           [0.5, 0.62],
    #                           [0.1, 0.2]]),
    #                  np.array([[1.83],
    #                            [-0.2],
    #                            [0.3]])
    #                  ]
    while True:
        MSE = 0
        for i in range(0, len(x)):
            net = forward_propagation(x[i], layers_weight, bias_flag)
            sigmas = back_propagation(net, layers_weight, y[0, i], bias_flag)
            layers_weight = update_weights(x[i], net, layers_weight, sigmas)
            net = forward_propagation(x[i], layers_weight, bias_flag)
            error = 0.5 * ((y[0, i] - net[-1][-1, -1]) ** 2)
            MSE = MSE + error
        MSE = MSE / len(x)
        if MSE <= threshold:
            break
    return layers_weight


layers_weight = train(x, bias_flag)
print(layers_weight)
print(forward_propagation(x[0], layers_weight, bias_flag)[-1])
print(forward_propagation(x[1], layers_weight, bias_flag)[-1])
print(forward_propagation(x[2], layers_weight, bias_flag)[-1])
print(forward_propagation(x[3], layers_weight, bias_flag)[-1])

plt.scatter([0, 0, 1, 1], [0, 1, 0, 1])
plt.grid(True)

axis11 = np.arange(0, 1.1, 0.1)
axis12 = (axis11 * layers_weight[0][1, 0] + layers_weight[0][0, 0]) / (-1 * layers_weight[0][2, 0])
plt.plot(axis11, axis12)

axis21 = np.arange(0, 1.1, 0.1)
axis22 = (axis21 * layers_weight[0][1, 1] + layers_weight[0][0, 1]) / (-1 * layers_weight[0][2, 1])
plt.plot(axis21, axis22)

# axis31 = np.arange(0, 1.1, 0.1)
# axis32 = (axis31 * layers_weight[1][1, 0] + layers_weight[1][0, 0]) / (-1 * layers_weight[1][2, 0])
# plt.plot(axis31, axis32)

plt.show()
