import numpy as np
import matplotlib.pyplot as plt

def load_data(file_path="./data/data_train.dt"):
    """
    Arguments:
        file_path: path to data set
    """
    data = np.loadtxt(file_path, delimiter=" ", dtype=float)
    X, y = data[:, 0].reshape(-1, 1), data[:, 1:].reshape(-1, 1)
    X = compute_padding(X)
    return X, y

def compute_padding(X):
    """
    Arguments:
        X: input
    Returns:
        augmented X with ones
    """
    ones = np.ones((X.shape[0], 1))
    X = np.concatenate((X, ones), axis=1)
    return X