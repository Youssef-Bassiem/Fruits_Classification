# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Ui_MainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1027, 825)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName("gridLayout_2")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem, 0, 0, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem1, 0, 2, 1, 1)
        self.output_widget = QtWidgets.QWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.output_widget.sizePolicy().hasHeightForWidth())
        self.output_widget.setSizePolicy(sizePolicy)
        self.output_widget.setObjectName("output_widget")
        self.output_horizontalLayout = QtWidgets.QHBoxLayout(self.output_widget)
        self.output_horizontalLayout.setObjectName("output_horizontalLayout")
        self.gridLayout_2.addWidget(self.output_widget, 1, 0, 1, 3)
        self.control_widget = QtWidgets.QWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.control_widget.sizePolicy().hasHeightForWidth())
        self.control_widget.setSizePolicy(sizePolicy)
        self.control_widget.setMinimumSize(QtCore.QSize(740, 0))
        self.control_widget.setObjectName("control_widget")
        self.gridLayout = QtWidgets.QGridLayout(self.control_widget)
        self.gridLayout.setSpacing(30)
        self.gridLayout.setObjectName("gridLayout")
        self.perceptron_btn = QtWidgets.QRadioButton(self.control_widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.perceptron_btn.sizePolicy().hasHeightForWidth())
        self.perceptron_btn.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.perceptron_btn.setFont(font)
        self.perceptron_btn.setObjectName("perceptron_btn")
        self.gridLayout.addWidget(self.perceptron_btn, 0, 2, 1, 2)
        self.label_3 = QtWidgets.QLabel(self.control_widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_3.sizePolicy().hasHeightForWidth())
        self.label_3.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 3, 0, 1, 1)
        self.epochs_spinBox = QtWidgets.QSpinBox(self.control_widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.epochs_spinBox.sizePolicy().hasHeightForWidth())
        self.epochs_spinBox.setSizePolicy(sizePolicy)
        self.epochs_spinBox.setMaximum(1000000)
        self.epochs_spinBox.setObjectName("epochs_spinBox")
        self.gridLayout.addWidget(self.epochs_spinBox, 3, 1, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.control_widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_4.sizePolicy().hasHeightForWidth())
        self.label_4.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 3, 2, 1, 1)
        self.label = QtWidgets.QLabel(self.control_widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 1, 0, 1, 1)
        self.bias_checkBox = QtWidgets.QCheckBox(self.control_widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.bias_checkBox.sizePolicy().hasHeightForWidth())
        self.bias_checkBox.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.bias_checkBox.setFont(font)
        self.bias_checkBox.setObjectName("bias_checkBox")
        self.gridLayout.addWidget(self.bias_checkBox, 5, 0, 1, 1)
        self.learningRate_spingBox = QtWidgets.QDoubleSpinBox(self.control_widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.learningRate_spingBox.sizePolicy().hasHeightForWidth())
        self.learningRate_spingBox.setSizePolicy(sizePolicy)
        self.learningRate_spingBox.setDecimals(6)
        self.learningRate_spingBox.setMaximum(100000000.0)
        self.learningRate_spingBox.setObjectName("learningRate_spingBox")
        self.gridLayout.addWidget(self.learningRate_spingBox, 1, 3, 1, 1)
        self.MSEThreshold_spinBox = QtWidgets.QDoubleSpinBox(self.control_widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.MSEThreshold_spinBox.sizePolicy().hasHeightForWidth())
        self.MSEThreshold_spinBox.setSizePolicy(sizePolicy)
        self.MSEThreshold_spinBox.setDecimals(6)
        self.MSEThreshold_spinBox.setMaximum(100000000.0)
        self.MSEThreshold_spinBox.setObjectName("MSEThreshold_spinBox")
        self.gridLayout.addWidget(self.MSEThreshold_spinBox, 3, 3, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.control_widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 1, 2, 1, 1)
        self.adaline_btn = QtWidgets.QRadioButton(self.control_widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.adaline_btn.sizePolicy().hasHeightForWidth())
        self.adaline_btn.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.adaline_btn.setFont(font)
        self.adaline_btn.setObjectName("adaline_btn")
        self.gridLayout.addWidget(self.adaline_btn, 0, 0, 1, 2)
        self.classes_comboBox = QtWidgets.QComboBox(self.control_widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.classes_comboBox.sizePolicy().hasHeightForWidth())
        self.classes_comboBox.setSizePolicy(sizePolicy)
        self.classes_comboBox.setMinimumSize(QtCore.QSize(150, 0))
        self.classes_comboBox.setObjectName("classes_comboBox")
        self.classes_comboBox.addItem("")
        self.classes_comboBox.addItem("")
        self.classes_comboBox.addItem("")
        self.gridLayout.addWidget(self.classes_comboBox, 1, 1, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.control_widget)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 4, 0, 1, 1)
        self.label_6 = QtWidgets.QLabel(self.control_widget)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.gridLayout.addWidget(self.label_6, 4, 2, 1, 1)
        self.comboBox = QtWidgets.QComboBox(self.control_widget)
        self.comboBox.setObjectName("comboBox")
        self.gridLayout.addWidget(self.comboBox, 4, 1, 1, 1)
        self.comboBox_2 = QtWidgets.QComboBox(self.control_widget)
        self.comboBox_2.setObjectName("comboBox_2")
        self.gridLayout.addWidget(self.comboBox_2, 4, 3, 1, 1)
        self.learn_btn = QtWidgets.QPushButton(self.control_widget)
        self.learn_btn.setObjectName("learn_btn")
        self.gridLayout.addWidget(self.learn_btn, 5, 3, 1, 1)
        self.gridLayout_2.addWidget(self.control_widget, 0, 1, 1, 1)
        self.gridLayout_2.setRowStretch(0, 1)
        self.gridLayout_2.setRowStretch(1, 3)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.perceptron_btn.setText(_translate("MainWindow", "Perceptron Algorithm"))
        self.label_3.setText(_translate("MainWindow", "Number of Epochs :"))
        self.label_4.setText(_translate("MainWindow", "MSE Threshold :"))
        self.label.setText(_translate("MainWindow", "Select Two Classes :"))
        self.bias_checkBox.setText(_translate("MainWindow", "Add Bias"))
        self.label_2.setText(_translate("MainWindow", "Learning Rate :"))
        self.adaline_btn.setText(_translate("MainWindow", "Adaline Algorithm"))
        self.classes_comboBox.setItemText(0, _translate("MainWindow", "BOMBAY & CALI"))
        self.classes_comboBox.setItemText(1, _translate("MainWindow", "BOMBAY & SIRA"))
        self.classes_comboBox.setItemText(2, _translate("MainWindow", "CALI & SIRA"))
        self.label_5.setText(_translate("MainWindow", "Feature 1 :"))
        self.label_6.setText(_translate("MainWindow", "Feature 2 :"))
        self.learn_btn.setText(_translate("MainWindow", "Run Algorithm"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
