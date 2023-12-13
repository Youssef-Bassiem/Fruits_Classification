import numpy as np
from PyQt5 import QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from Task_1.Perceptron_Algorithm import main
from Task_1.Adaline_Algorithms import main_adaline

from Ui_MainWindow import Ui_MainWindow


class MainWindow(Ui_MainWindow):
    def __init__(self):
        self.win = QtWidgets.QMainWindow()
        self.setupUi(self.win)
        self.win.show()

        self.comboBox.addItems(['Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'roundnes'])
        self.comboBox_2.addItems(['Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'roundnes'])
        self.learn_btn.clicked.connect(lambda: self.deep_learning())

        # adding Graph
        self.graph = FigureCanvasQTAgg(Figure(figsize=(1, 1), dpi=100))
        self.output_horizontalLayout.addWidget(self.graph)
        self.graph_axis = self.graph.figure.add_subplot(111)
        # self.graph_axis.grid(True)

        # default values
        self.perceptron_btn.setChecked(True)
        self.classes_comboBox.setCurrentIndex(0)
        self.epochs_spinBox.setValue(150)
        self.learningRate_spingBox.setValue(0.0001)
        self.bias_checkBox.setChecked(True)

    def deep_learning(self):
        # 0 -> Bombay, 1 -> Sira, 2 -> Cali
        if self.classes_comboBox.currentText() == 'BOMBAY & CALI':
            main.C1 = 0
            main.C2 = 2
        elif self.classes_comboBox.currentText() == 'BOMBAY & SIRA':
            main.C1 = 0
            main.C2 = 1
        elif self.classes_comboBox.currentText() == 'CALI & SIRA':
            main.C1 = 2
            main.C2 = 1
        # print(self.classes_comboBox.currentText())

        main.flagOfBias = False
        if self.bias_checkBox.isChecked():
            main.flagOfBias = True

        main.features = ['Class', self.comboBox.currentText(), self.comboBox_2.currentText()]
        main.epochs = self.epochs_spinBox.value()
        main.L = self.learningRate_spingBox.value()

        data1, data2, weights = main.main()

        self.graph_axis.clear()
        self.graph.draw()
        # print(data2[self.comboBox.currentText()].tolist())
        self.graph_axis.scatter(data1[self.comboBox.currentText()].tolist(),
                                data1[self.comboBox_2.currentText()].tolist())
        self.graph_axis.scatter(data2[self.comboBox.currentText()].tolist(),
                                data2[self.comboBox_2.currentText()].tolist())

        y = np.linspace(0,
                        max(data1[self.comboBox_2.currentText()].max(), data2[self.comboBox_2.currentText()].max()),
                        10)
        x = ((weights[2] * y) + weights[0]) / (-1 * weights[1])

        self.graph_axis.plot(x, y, color='red')
        self.graph_axis.grid(True)
        self.graph.draw()

    def deep_learning_2(self):
        # 0 -> Bombay, 1 -> Sira, 2 -> Cali
        if self.classes_comboBox.currentText() == 'BOMBAY & CALI':
            main_adaline.C1 = 0
            main_adaline.C2 = 2
        elif self.classes_comboBox.currentText() == 'BOMBAY & SIRA':
            main_adaline.C1 = 0
            main_adaline.C2 = 1
        elif self.classes_comboBox.currentText() == 'CALI & SIRA':
            main_adaline.C1 = 2
            main_adaline.C2 = 1

        main_adaline.flagOfBias = False
        if self.bias_checkBox.isChecked():
            main_adaline.flagOfBias = True

        main_adaline.features = ['Class', self.comboBox.currentText(), self.comboBox_2.currentText()]
        main_adaline.epochs = self.epochs_spinBox.value()
        main_adaline.L = self.learningRate_spingBox.value()
        main_adaline.threshold = self.MSEThreshold_spinBox.value()
        data1, data2, weights = main_adaline.main()

        self.graph_axis.clear()
        self.graph.draw()
        self.graph_axis.scatter(data1[self.comboBox.currentText()].tolist(),
                                data1[self.comboBox_2.currentText()].tolist())
        self.graph_axis.scatter(data2[self.comboBox.currentText()].tolist(),
                                data2[self.comboBox_2.currentText()].tolist())

        y = np.linspace(0,
                        max(data1[self.comboBox_2.currentText()].max(), data2[self.comboBox_2.currentText()].max()),
                        10)
        x = ((weights[2] * y) + weights[0]) / (-1 * weights[1])
        # x += 70000
        # y += 800
        self.graph_axis.plot(x, y, color='red')
        self.graph_axis.grid(True)
        self.graph.draw()


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    ui = MainWindow()
    sys.exit(app.exec_())
