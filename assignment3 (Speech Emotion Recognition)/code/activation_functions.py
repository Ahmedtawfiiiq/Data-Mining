import numpy as np


# sigmoid activation function (used for binary classification)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# softmax activation function (used for multi-class classification)
def softmax_stable(x):
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps)


# ReLU activation function (used for hidden layers)
def relu(x):
    return np.maximum(0, x)


# derivative of ReLU activation function
def relu_derivative(x):
    return np.where(x > 0, 1, 0)
