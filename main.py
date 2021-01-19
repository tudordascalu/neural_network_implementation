import numpy as np
import matplotlib.pyplot as plt
import sys
from utils import neural_net, db

# Check for arguments
arguments = sys.argv
has_arguments = len(arguments) > 1
# Load data
X_train, y_train = db.load_data("./data/data_train.dt")
X_val, y_val = db.load_data("./data/data_val.dt")
# Normalize data
X_mean = np.mean(X_train, axis=0)
X_std = np.std(X_train, axis=0)
X_train = db.compute_normalization(X_train, X_mean, X_std)
X_val = db.compute_normalization(X_val, X_mean, X_std)
# Pad data
X_train, X_val = db.compute_padding(X_train), db.compute_padding(X_val)
# Start application
if not has_arguments or "0" in arguments:
    """
    Visualize data set
    """
    plt.figure(figsize=(10, 8))
    plt.scatter(X_train[:, 0], y_train, c="b", label="Train")
    plt.scatter(X_val[:, 0], y_val, c="r", label="Val")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()
    
if not has_arguments or "1" in arguments:
    """
    Train neural network and save model to file
    """
    nn = neural_net.Neural_net(n_dim=1, n_neurons=20, learning_rate=1e-3)
    nn.fit(X_train, y_train, batch_size=2, n_iter=20000, X_val=X_val, y_val=y_val, verbose=True)

if "2" in arguments:
    data = [{"a": np.array([1, 2, 3])}, {"a": np.array([1, 2, 3, 4])}]
    data = [el for d in data for el in d["a"]]
    print(data)