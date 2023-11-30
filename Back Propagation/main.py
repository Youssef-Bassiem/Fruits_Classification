from MultiNeuralNetwork import MultiNeuralNetwork
import PreProcessing
import pandas as pd


x = pd.DataFrame({
    'x1': [0, 1, 1, 0],
    'x2': [0, 1, 0, 1]
})
y = pd.DataFrame({'y': [0, 0, 1, 1]})

model = MultiNeuralNetwork([5],
                           True,
                           0.1,
                           500,
                           MultiNeuralNetwork.sigmoid,
                           MultiNeuralNetwork.sigmoid_derivative
                           )
x_train, y_train, x_test, y_test = PreProcessing.main()
layers_weights = model.train(x_train, y_train)
y_pred = model.test(x_test, layers_weights)

print('train accuracy : ', model.accuracy(x_train, y_train, layers_weights))
print('test accuracy : ', model.accuracy(x_test, y_test, layers_weights))
print("*****************************************************************************")
print("-------Confusion Matrix-------")
# print(confusion_matrix(y_test, y_pred))
# print("++++++++++++++++++++++++++++++++")
model.plot_confusion_matrix(model.confusion_matrix(y_test, y_pred), ["BOMBAY", "CALI", "SIRA"])
print("*****************************************************************************")