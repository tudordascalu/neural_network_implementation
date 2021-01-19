"""
Neural network
    - handles 1 input layer with alternative sigmoid activation function
"""
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
                "partial_derivatives": [],
                "z": None,  # Linear response = w0 + w1a1 + w2a2
                "a": None # Activation response = sigmoid(z)
            },
            {
                "type": "output",
                "weights": np.random.random((1, n_neurons + 1)), # Set of weights correspondnig to all neurons
                "n_neurons": n_neurons,
                "delta": None, # Derivative of error with respect to the activation a for each neuron
                "partial_derivatives": [],
                "z": None, # Linear response = w0 + w1a1 + w2a2
                "a": None, # Activation response = sigmoid(z)
            }
        ]
        self.n_layers = len(self.network) # Lenght of network not counting input layer
        self.learning_rate = learning_rate

    def f_sigmoid(self, X): # TODO: rename to appropriate funtion name
        """
        Arguments:
            X: input
        Returns
            f(X) = \frac{X}{1 + |X|}
        """
        return X / (1 + np.abs(X))

    def f_relu(self, X):
        """
        Arguments:
            X: input
        Returns:
            max(0, X)
        """
        return np.where(X >= 0, X, 0)
    
    def df_relu_dx(self, X):
        """
        Arguments:
            X: input
        Returns:
            max(0, 1)
        """
        return np.where(X >= 0, 1, 0)

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

    def compute_batches(self, X, y, batch_size):
        """
        Arguments:
            X: input
            y: labels
            batch_size: number of elements in each batch
        Returns:
            X_batches
        """
        i_shuffle = np.arange(0, X.shape[0])
        np.random.shuffle(i_shuffle)
        X = X[i_shuffle, :]
        y = y[i_shuffle, :]
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
        for i_layer in range(self.n_layers)[::-1]: # Propagate error through network
            self.network[i_layer]["partial_derivatives"] = []
            self.network[i_layer]["delta"] = []
            if self.network[i_layer]["type"] == "output":
                delta = self.network[i_layer]["a"].reshape(-1, 1) - y # Derivtive of MSE w.r.t (y_pred - y) for each training example
                self.network[i_layer]["delta"].append(delta)
            else:
                for i_neuron in range(self.network[i_layer]["n_neurons"]):
                    delta = self.network[i_layer + 1]["delta"][0] * self.network[i_layer + 1]["weights"][:, i_neuron] * self.df_sigmoid_dx(self.network[i_layer]["z"][i_neuron].reshape(-1, 1))
                    self.network[i_layer]["delta"].append(delta)
                    
        for i_layer in range(self.n_layers - 1): # Update weights
            for i_neuron in range(self.network[i_layer]["n_neurons"]):
                partial_derivatives = np.sum(self.network[i_layer]["delta"][i_neuron] * self.network[i_layer]["input"].T, axis=0)
                self.network[i_layer]["weights"][i_neuron] -= self.learning_rate * partial_derivatives
                self.network[i_layer]["partial_derivatives"].append(partial_derivatives)


    def fit(self, X, y, n_epochs=500, threshold_gradient=1e-10, threshold_bad=10, batch_size=5, X_val=[], y_val=[], verbose=False):
        """
        Arguments:
            X: input; shape(n, n_dim)
            y: labels; shape=(n, 1)
            n_epochs: maximum number of iterations before stopping
            threshold: minimum gradient magnitude threshold
            X_val: validation input; used in case log is true
            y_val: validation labels; used in case log is true
        Returns:
            updates the weights of each layer
        """
        error_val_list = []
        error_train_list = []
        count_bad = 0
        error_val_past = np.inf
        for i_epoch in range(n_epochs): # Loop through epochs
            X_batches, y_batches = self.compute_batches(X, y, batch_size) # Split data set
            for i_batch, X_batch in enumerate(X_batches): # Iterate through all batches
                # Forward pass
                y_pred = self.compute_forwardpropagation(X_batch) 
                # Compute loss
                error_train = self.f_mse(y_batches[i_batch], y_pred)
                error_train_list.append(error_train)
                # Backpropagate errors
                self.compute_backpropagation(X, y_batches[i_batch])
                # Compute gradient magnitude
                partial_derivatives = [partial_derivative for layer in self.network for partial_derivative in layer["partial_derivatives"]]
                gradient_magnitude = np.sum(np.power(partial_derivatives, 2))
                # Validate model
                error_val = self.f_mse(self.predict(X_val), y_val)
                error_val_list.append(error_val)
                # Log data
                if verbose == True:
                    print("----")
                    print("Iteration: {}".format(i_epoch))
                    print("Gradient magnitude: {}".format(gradient_magnitude))
                    print("Train error: {}".format(error_train))
                    print("Validation error: {}".format(error_val))
                # Early stopping
                if error_val_past < error_val: 
                    count_bad += 1
                    if count_bad > threshold_bad: 
                        print("More than {} iterations in a row with worse validation errors".format(count_bad))
                        return error_train_list, error_val_list
                else: 
                    count_bad = 0
                if gradient_magnitude < threshold_gradient: 
                    print("Gradient magnitude {} is below threshold {}".format(gradient_magnitude, threshold_gradient))
                    return error_train_list, error_val_list
                error_val_past = error_val
        return error_train_list, error_val_list

    def predict(self, X):
        """
        Arguments:
            X: input
        Returns:
            y_pred
        """
        return self.compute_forwardpropagation(X)
    
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

        

        