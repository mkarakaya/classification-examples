import numpy as np
import pandas as pd


def get_data():
    df = pd.read_csv('dataset/winequality-red.csv', sep=';')
    data = df.values
    np.random.shuffle(data)
    percentage = int(len(data) * 0.8)
    training, test = data[:percentage, :], data[percentage:, :]
    training_labels = training[:, training.shape[1]-1:]
    test_labels = test[:, test.shape[1]-1:]
    return training[:, :training.shape[1]-1], training_labels, test[:, :test.shape[1]-1], test_labels


train_X, train_Y, test_X, test_Y = get_data()
print(train_X.shape, ':', train_Y.shape, ':', test_X.shape, ':', test_Y.shape)