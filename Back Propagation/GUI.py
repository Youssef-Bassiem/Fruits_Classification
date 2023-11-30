import numpy as np
from PyQt5 import QtWidgets
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from sklearn.metrics import confusion_matrix
import seaborn as sns
from Ui_MainWindow import Ui_MainWindow
from MultiNeuralNetwork import MultiNeuralNetwork
import PreProcessing


class GUI(Ui_MainWindow):
    def __init__(self):
        self.win = QtWidgets.QMainWindow()
        self.setupUi(self.win)

        self.last_layers_num = self.spinBox.value()
        self.spinBox.valueChanged.connect(lambda: self.fun1())
        self.learn_btn.clicked.connect(lambda: self.run_algorithm())

        self.epochs_spinBox.setValue(500)
        self.learningRate_spingBox.setValue(0.1)
        self.sigmoid_btn.toggle()
        self.bias_checkBox.setChecked(True)

    def fun1(self):
        if self.spinBox.value() > self.last_layers_num:
            s = QtWidgets.QSpinBox(self.widget_2)
            s.setMinimum(1)
            s.setMaximum(1000000)
            self.horizontalLayout.addWidget(s)
            self.last_layers_num = self.spinBox.value()
        elif self.spinBox.value() < self.last_layers_num:
            self.horizontalLayout.itemAt(0).widget().setParent(None)
            self.last_layers_num = self.spinBox.value()

    def get_layers_neurons(self):
        layers_neurons = []
        for i in range(self.horizontalLayout.count()):
            layers_neurons.append(self.horizontalLayout.itemAt(i).widget().value())
        return layers_neurons

    def run_algorithm(self):
        layers_neurons = self.get_layers_neurons()
        activ_func = MultiNeuralNetwork.sigmoid
        activ_func_deriv = MultiNeuralNetwork.sigmoid_derivative
        if self.Tanh_btn.isChecked():
            activ_func = MultiNeuralNetwork.tanh
            activ_func_deriv = MultiNeuralNetwork.tanh_derivative

        model = MultiNeuralNetwork(layers_neurons,
                                   self.bias_checkBox.isChecked(),
                                   self.learningRate_spingBox.value(),
                                   self.epochs_spinBox.value(),
                                   activ_func,
                                   activ_func_deriv
                                   )
        x_train, y_train, x_test, y_test = PreProcessing.main()
        layers_weights = model.train(x_train, y_train)
        y_pred = model.test(x_test, layers_weights)

        print('train accuracy : ', model.accuracy(x_train, y_train, layers_weights))
        print('test accuracy : ', model.accuracy(x_test, y_test, layers_weights))
        print("*****************************************************************************")
        print("-------Confusion Matrix-------")
        self.plot_confusion_matrix(model.confusion_matrix(y_test, y_pred), ["BOMBAY", "CALI", "SIRA"])
        print("*****************************************************************************")

    def plot_confusion_matrix(self, conf_matrix, class_names):
        if self.output_horizontalLayout.count() > 0:
            self.output_horizontalLayout.itemAt(0).widget().setParent(None)
        self.graph = FigureCanvasQTAgg(Figure(figsize=(1, 1), dpi=100))
        self.graph_axis = self.graph.figure.add_subplot(111)
        self.output_horizontalLayout.addWidget(self.graph)

        sns.heatmap(conf_matrix, ax=self.graph_axis, annot=True, fmt="d",
                    xticklabels=class_names, yticklabels=class_names)

        self.graph.draw()





if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    gui = GUI()
    gui.win.show()
    sys.exit(app.exec_())