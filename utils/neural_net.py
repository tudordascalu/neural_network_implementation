import numpy as np
import matplotlib.pyplot as plt

class Neural_net():
    """
    Neural network model with one hidden layer
    """
    def __init__(self, n_dim, n_neurons, learning_rate):
        """
        Arguments:
            n_dim: number of dimensions of input
            n_neurons: number of neurons in hidden layer
        """
        assert len(n_neurons) == n_layers # Check that the number of neurons per layer is sepcified
        self.layers = [
            {
                "layer": "hidden",
                "weights": np.zeros((n_neurons, n_dim + 1)),
                "activation": self.f,
                "dactivation_dx": self.df_dx,
                "gradient": None,
                "output": None
            },
            {
                "layer": "output",
                "weights": np.zeros((1, n_neurons)),
                "activation": None,
                "gradient": None,
                "output": None
            }
        ]
        self.learning_rate = learning_rate

    def f(self, X):
        """
        Arguments:
            X: list of neuron outputs
        Returns
            f(X) = \frac{X}{1 + |X|}
        """
        return X / (1 + np.abs(X))

    def df_dx(self, X):
        """
        Arguments:
            X: list of neuron outputs
        Returns:
            \frac{df}{dx}  = \frac{1}{(1 + |X|)^2}
        """
        return 1 / (1 + np.abs(X))**2

    def compute_batches(self, X, y, batch_size):
        """
        Arguments:
            X: input
            y: labels
            batch_size: number of elements in each batch
        Returns:
            X_batches
        """
        return np.split(X, batch_size, axis=0), np.split(y, batch_size, axis=0)

    def compute_forward_pass(self, X):
        """
        Arguments:
            X: input
        Returns:
            void; sets output on each layer
        """
        for i, layer in enumerate(self.layers):
            output = layer.weights @ X.T # Compute linear activation
            if layer.activation != None:
                output = layer.activation(output)
            self.layers[i].output = output

    def fit(self, X, y, n_iter=500, threshold=1e-3, batch_size):
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
        for i in range(n_iter): # Loop through iterations
            for X_batch in X_batches: # Loop through batches
                # Forward pass
                self.compute_forward_pass(X_batch)
                print(self.layers)
                # Loss computation
                # Computation of gradients
                # Check if gradient magnitude is smaller than threshold
                # Update weights

        

        