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

if not has_arguments or "0" in arguments:
    """
    Visualize data set
    """
    plt.figure(figsize=(10, 8))
    plt.scatter(X_train, y_train, c="b", label="Train")
    plt.scatter(X_val, y_val, c="r", label="Val")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()
    
if not has_arguments or "1" in arguments:
    """
    Train neural network and save model to file
    """
    nn = neural_net.Neural_net(n_dim=1, n_neurons=3, learning_rate=1e-5)
    nn.fit(X_train, y_train, n_iter=500)

if "2" in arguments:
    pass