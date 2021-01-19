# Project description
- self implementation of a single hidden layer neural network with varying number of neurons
- the hidden layer uses alternative sigmoid as activation function
- the output layer includes one neuron
- the loss function is MSE
# Data
- the data set is synthetic, created with the function $f(x) = \frac{sin(x)}{x}$
- it is standardized to 0 mean and unit variance

# Running the projet
- running all parts of the project:
```
python3 main.py
```
- plotting the data:
```
python3 main.py 0
```
- training the neural network
```
python3 main.py 1
```

# Dependencies
- numpy
- matplotlib

# Possible improevements
- enable multiple hidden layers
- enable mutliple activation functions
- add shortcut connections