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
