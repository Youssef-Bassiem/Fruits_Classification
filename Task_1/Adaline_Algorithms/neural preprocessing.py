
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,MinMaxScaler 
from sklearn.model_selection import train_test_split


data =pd.read_csv("/Task_1/Perceptron_Algorithm/Dry_Bean_Dataset.xlsx")

print("-------------------------")

print(data.columns)

print("-------------------------")
print(data['Class'].head)
print(data.isna().sum())
data['MinorAxisLength'] = data['MinorAxisLength'].fillna(data.MinorAxisLength.median())
print(data.isna().sum())

x=data.drop('Class',axis=1)
y=data['Class']

x_train , x_test, y_train ,y_test=train_test_split(x,y,test_size=0.40)

print(x_train)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

labelencoder=LabelEncoder()
y_train['Class']=labelencoder.fit_transform(y_train)
print(y_train)
minmax=MinMaxScaler(feature_range=(0,1))
x_train=minmax.fit_transform(x_train)
print("-------------------------")
print(x_train[0])

#algorthim
