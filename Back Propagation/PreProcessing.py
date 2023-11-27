import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


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
        data_classes[int(row['Class'])] = data_classes[int(row['Class'])]._append(row, ignore_index=True)

    train_data = pd.DataFrame(columns=df.columns)
    test_data = pd.DataFrame(columns=df.columns)
    for j in range(len(data_classes)):
        class_train, class_test = train_test_split(data_classes[j], test_size=0.4, random_state=42)
        train_data = pd.concat([train_data, class_train], ignore_index=True)
        test_data = pd.concat([test_data, class_test], ignore_index=True)

    # scale data
    columns_to_scale = ['Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'roundnes']
    # print(train_data)
    for column in columns_to_scale:
        scale_and_replace(train_data, column, -1, 1)
        scale_and_replace(test_data, column, -1, 1)

    y_train = pd.DataFrame(train_data['Class'])
    train_data.drop(['Class'], axis=1, inplace=True)
    y_test = pd.DataFrame(test_data['Class'])
    test_data.drop(['Class'], axis=1, inplace=True)
    return train_data, y_train, test_data, y_test


def scale_and_replace(df, column, a, b):
    values = np.array(df[column]).reshape(-1, 1)

    # Feature scaling
    scaler = MinMaxScaler(feature_range=(a, b))
    scaled_values = scaler.fit_transform(values)

    # Replace zero values with the median
    median_value = np.median(scaled_values[scaled_values != 0])
    scaled_values[scaled_values == 0] = median_value

    # Assign the scaled and replaced values back to the DataFrame
    df[column] = scaled_values.flatten()



if __name__ == '__main__':
    main()