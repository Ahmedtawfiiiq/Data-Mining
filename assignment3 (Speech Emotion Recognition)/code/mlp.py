from activation_functions import relu, relu_derivative, softmax_stable
import numpy as np


# MLP Training: Stochastic Gradient Descent
# single hidden layer
class MLP:
    def __init__(self, dataset, labels, lr, n_classes, hidden_neurons):
        self.dataset = dataset
        self.labels = labels
        self.learning_rate = lr
        self.wh = np.random.randn(hidden_neurons, len(dataset[0])) - 0.5
        self.bh = (
            np.random.randn(
                hidden_neurons,
            )
            - 0.5
        )
        self.wo = np.random.randn(n_classes, hidden_neurons) - 0.5
        self.bo = (
            np.random.randn(
                n_classes,
            )
            - 0.5
        )

    def forward_propagation(self):  # feed forward phase
        self.z = relu(np.dot(self.x, self.wh.T) + self.bh)
        self.o = softmax_stable(np.dot(self.z, self.wo.T) + self.bo)

    def backpropagation(self):  # backpropagation phase
        # softmax derivative
        delta_o = self.o - self.y
        # ReLU derivative
        delta_h = np.multiply(relu_derivative(self.z), np.dot(delta_o, self.wo))
        # gradient descent for bias vectors
        d_bo = delta_o
        self.bo = self.bo - self.learning_rate * d_bo
        d_bh = delta_h
        self.bh = self.bh - self.learning_rate * d_bh
        # gradient descent for weight matrices
        d_wo = np.dot(delta_o.T, self.z)
        self.wo = self.wo - self.learning_rate * d_wo
        d_wh = np.dot(delta_h.T, self.x)
        self.wh = self.wh - self.learning_rate * d_wh

    def train(self, maxiter):
        t = 0
        while True:
            for j in range(len(self.dataset)):
                self.x = np.array([self.dataset[j]])
                self.y = self.labels[j]
                self.forward_propagation()
                self.backpropagation()
            t += 1
            if t == maxiter:
                break

    def predict(self, test_dataset, test_labels):
        count = 0
        for i in range(len(test_dataset)):
            self.x = np.array([test_dataset[i]])
            self.y = test_labels[i]
            self.forward_propagation()
            # print("output:\n", self.o)
            # print("sum of output:\n", np.sum(self.o))
            if np.argmax(self.o) == np.argmax(self.y):
                # print("prediction", np.argmax(self.o))
                count += 1
            # else:
            # print("prediction:", np.argmax(self.o), "true:", np.argmax(self.y))
        return (count / len(test_dataset)) * 100
