import pandas as pd
import numpy as np

from sklearn import preprocessing


def shuffle_data(start, all_data, samples, features):
    return (all_data.loc[start: start + samples - 1, features].
            sample(frac=1, random_state=42)).reset_index(drop=True)


def normalize(data):
    cols = data.columns
    x = data.values  # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled)
    df.columns = cols
    return df


def split_data(c, samples, features, train_samples, df):
    start_of_c = c * samples
    data_of_c = shuffle_data(start_of_c, df, samples, features)
    c1_class = data_of_c['Class']
    data_of_c.drop(['Class'], axis=1, inplace=True)
    data_of_c.insert(0, 'Class', c1_class.tolist())

    samples_of_c = data_of_c.loc[0: train_samples - 1, features]
    test_of_c = data_of_c.loc[train_samples: samples, features]
    return samples_of_c, test_of_c


def pre(c1, c2, samples, train_samples, features):
    df = pd.read_excel("../Dry_Bean_Dataset.xlsx")
    samples_of_c1, test_of_c1 = split_data(c1, samples, features, train_samples, df)
    samples_of_c2, test_of_c2 = split_data(c2, samples, features, train_samples, df)

    samples_of_c1[samples_of_c1.columns[0]] = np.ones(samples_of_c1.shape[0])
    test_of_c1[test_of_c1.columns[0]] = np.ones(test_of_c1.shape[0])

    samples_of_c2[samples_of_c2.columns[0]] = np.ones(samples_of_c2.shape[0]) * -1
    test_of_c2[test_of_c2.columns[0]] = np.ones(test_of_c2.shape[0]) * -1

    data = (pd.concat([samples_of_c1, samples_of_c2]).
            sample(frac=1, random_state=42).reset_index(drop=True))

    for feature in features:
        data[feature].fillna(value=data[feature].median(), inplace=True)
        test_of_c1[feature].fillna(value=test_of_c1[feature].median(), inplace=True)
        test_of_c2[feature].fillna(value=test_of_c2[feature].median(), inplace=True)
    data.iloc[:, 1:len(features)] = normalize(data.iloc[:, 1:len(features)])
    test_of_c1.drop(['Class'], axis=1, inplace=True)
    test_of_c2.drop(['Class'], axis=1, inplace=True)
    test_of_c1 = normalize(test_of_c1)
    test_of_c2 = normalize(test_of_c2)
    test_of_c1.insert(0, 'Class', np.ones(test_of_c1.shape[0]))
    test_of_c2.insert(0, 'Class', np.ones(test_of_c2.shape[0])*-1)
    dd1 = pd.DataFrame()
    dd2 = pd.DataFrame()
    for i in range(data.shape[0]):
        if data.iloc[i, 0] == 1:
            dd1 = dd1.append(data.iloc[i, 1:], ignore_index=True)
        else:
            dd2 = dd2.append(data.iloc[i, 1:], ignore_index=True)
    return dd1, dd2, data, samples_of_c1, samples_of_c2, test_of_c1, test_of_c2
