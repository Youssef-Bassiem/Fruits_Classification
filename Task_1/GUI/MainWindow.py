from PyQt5 import QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from Task_1.Perceptron_Algorithm import main
from Ui_MainWindow import Ui_MainWindow


class MainWindow(Ui_MainWindow):
    def __init__(self):
        self.features = ['Class', 'Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'roundnes']
        self.win = QtWidgets.QMainWindow()
        self.setupUi(self.win)
        self.win.show()

        self.checkBox.clicked.connect(lambda: self.load_features('Area', self.checkBox))
        self.checkBox_2.clicked.connect(lambda: self.load_features('Perimeter', self.checkBox_2))
        self.checkBox_3.clicked.connect(lambda: self.load_features('MajorAxisLength', self.checkBox_3))
        self.checkBox_4.clicked.connect(lambda: self.load_features('MinorAxisLength', self.checkBox_4))
        self.checkBox_5.clicked.connect(lambda: self.load_features('roundnes', self.checkBox_5))
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
        self.checkBox.setChecked(True)
        self.checkBox_2.setChecked(True)
        self.checkBox_3.setChecked(True)
        self.checkBox_4.setChecked(True)
        self.checkBox_5.setChecked(True)


    def load_features(self, feature_name, box):
        if box.isChecked():
            self.features.append(feature_name)
        if not box.isChecked():
            self.features.remove(feature_name)

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

        main.features = self.features
        print(main.features)
        main.epochs = self.epochs_spinBox.value()
        main.L = self.learningRate_spingBox.value()

        main.main()

        self.graph_axis.clear()
        self.graph.draw()
        self.graph_axis.scatter([0, 1, 2], [0, 1, 2])
        self.graph_axis.scatter([2, 2, 3], [1, 1, 2])
        # self.graph_axis.grid(True)
        self.graph.draw()


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    ui = MainWindow()
    sys.exit(app.exec_())
