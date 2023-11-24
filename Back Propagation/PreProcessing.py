import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    df = pd.read_excel("../Task_1/Dry_Bean_Dataset.xlsx")
    null_values = df.isnull().any()
    for column in null_values[null_values].index:
        mean_value = df[column].mean()
        df[column].fillna(mean_value, inplace=True)
    null_values = df.isnull().any()

    unique_counts = df.nunique()
    return fill_classes(df)


def fill_classes(df):
    unique_counts = df['Class'].unique()
    class_mapping = {cls: idx for idx, cls in enumerate(unique_counts)}
    df['Class'] = df['Class'].replace(class_mapping)
    data_classes = []
    for i in range(len(unique_counts)):
        data_classes.append(pd.DataFrame(columns=df.columns))
    for index, row in df.iterrows():
        data_classes[int(row['Class'])] = data_classes[int(row['Class'])].append(row, ignore_index=True)
    train_data = pd.DataFrame(columns=df.columns)
    test_data = pd.DataFrame(columns=df.columns)
    for j in range(len(data_classes)):
        class_train, class_test = train_test_split(data_classes[j], test_size=0.4, random_state=42)
        train_data = pd.concat([train_data, class_train], ignore_index=True)
        test_data = pd.concat([test_data, class_test], ignore_index=True)

    y_train = train_data['Class']
    x_train = train_data.drop(['Class'], axis=1, inplace=True)
    y_test = test_data['Class']
    x_test = test_data.drop(['Class'], axis=1, inplace=True)
    return x_train, y_train, x_test, y_test

if __name__ == '__main__':
    main()