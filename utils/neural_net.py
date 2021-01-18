import numpy as np
import matplotlib.pyplot as plt

class Neural_net():
    """
    Neural network model with one hidden layer
    """
    def __init__(self, n_dim, n_neurons, learning_rate=1e-5):
        """
        Arguments:
            n_dim: number of dimensions of input
            n_neurons: number of neurons in hidden layer
        """
        self.network = [
            {
                "type": "hidden",
                "weights": np.random.random((n_neurons, n_dim + 1)), # Set of weights corresponding to all neurons
                "n_neurons": n_neurons,
                "delta": None, # Derivative of error with respect to the activation a for each neuron
                "z": None,  # Linear response = w0 + w1a1 + w2a2
                "a": None # Activation response = sigmoid(z)
            },
            {
                "type": "output",
                "weights": np.random.random((1, n_neurons + 1)), # Set of weights correspondnig to all neurons
                "n_neurons": n_neurons,
                "delta": None, # Derivative of error with respect to the activation a for each neuron
                "z": None, # Linear response = w0 + w1a1 + w2a2
                "a": None, # Activation response = sigmoid(z)
            }
        ]
        self.n_layers = len(self.network) # Lenght of network not counting input layer
        self.learning_rate = learning_rate

    def f_sigmoid(self, X): # TODO: rename to appropriate funtion name
        """
        Arguments:
            X: list of neuron outputs
        Returns
            f(X) = \frac{X}{1 + |X|}
        """
        return X / (1 + np.abs(X))

    def df_sigmoid_dx(self, X):
        """
        Arguments:
            X: list of neuron outputs
        Returns:
            \frac{df}{dx}  = \frac{1}{(1 + |X|)^2}
        """
        return 1 / (1 + np.abs(X))**2

    def f_mse(self, y, y_pred):
        """
        Arguments:
            y: predicted labes
            y_pred: true labels
        Returns:
            MSE
        """
        return np.sum((y_pred - y)**2)

    def df_mse(self, y, y_pred):
        """
        Arguments:
            y: predicted labels
            y_pred: true labels
        Returns:
            derivative w.r.t (y_pred - y)
        """
        return 2 * (y_pred - y)

    def compute_batches(self, X, y, batch_size):
        """
        Arguments:
            X: input
            y: labels
            batch_size: number of elements in each batch
        Returns:
            X_batches
        """
        n_batches = int(X.shape[0] / batch_size)
        return np.array_split(X, n_batches, axis=0), np.array_split(y, n_batches, axis=0)

    def compute_forwardpropagation(self, X):
        """
        Arguments:
            X: input
        Returns:
            predictions on bactch
        """
        self.network[0]["input"] = X.T
        layer_input = X.T # First layer input
        for i_layer, layer in enumerate(self.network):
            z = layer["weights"] @ layer["input"] # Compute linear activation
            a = z
            if layer["type"] == "hidden":
                a = self.f_sigmoid(z)
            self.network[i_layer]["z"] = z
            self.network[i_layer]["a"] = a
            if layer["type"] != "output":
                self.network[i_layer + 1]["input"] = self.compute_padding(a.T).T # Compute input for next layer
        return a.reshape(-1, 1) # Predictions

    def compute_backpropagation(self, X, y):
        """
        Arguments:
            X: input
        Returns:
            gradient w.r.t weights
        """
        for i_layer in range(self.n_layers)[::-1]: # Iterate through network sarting at the end layer
            self.network[i_layer]["delta"] = []
            if self.network[i_layer]["type"] == "output":
                delta = np.sum(2 * (self.network[i_layer]["a"].reshape(-1, 1) - y))
                self.network[i_layer]["delta"].append(delta)
            else:
                for i_neuron in range(self.network[i_layer]["n_neurons"]):
                    delta = np.sum(self.network[i_layer + 1]["delta"] * self.network[i_layer + 1]["weights"][:, i_neuron] * self.df_sigmoid_dx(self.network[i_layer]["z"][i_neuron]))
                    self.network[i_layer]["delta"].append(delta)
                    
        for i_layer in range(self.n_layers - 1): # Update weights
            for i_neuron in range(self.network[i_layer]["n_neurons"]):
                self.network[i_layer]["weights"][i_neuron] -= np.sum(self.learning_rate * self.network[i_layer]["delta"][i_neuron] * self.network[i_layer]["input"], axis=1)


    def fit(self, X, y, n_iter=500, threshold=1e-3, batch_size=5):
        """
        Arguments:
            X: input; shape(n, n_dim)
            y: labels; shape=(n, 1)
            n_iter: maximum number of iterations before stopping
            threshold: minimum gradient magnitude threshold
        Returns:
            updates the weights of each layer
        """
        X_batches, y_batches = self.compute_batches(X, y, batch_size) # Split data set 
        for i_iter in range(n_iter): # Loop through iterations
            for i_batch, X_batch in enumerate(X_batches): # Loop through batches
                y_pred = self.compute_forwardpropagation(X_batch) # Forward pass
                loss = self.f_mse(y_batches[i_batch], y_pred) # Compute loss
                self.compute_backpropagation(X, y_batches[i_batch])
                print(loss)

    def compute_padding(self, X):
        """
        Arguments:
            X: input
        Returns:
            augmented X with ones
        """
        ones = np.ones((X.shape[0], 1))
        X = np.concatenate((X, ones), axis=1)
        return X

        

        