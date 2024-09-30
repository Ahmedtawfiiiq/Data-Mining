from activation_functions import relu, relu_derivative, softmax_stable
import numpy as np


# Deep MLP Training: Stochastic Gradient Descent
# multiple hidden layers
class DEEP_MLP:
    # handle multiple hidden layers
    def __init__(self, dataset, labels, lr, n_classes, hidden_neurons):
        self.dataset = dataset
        self.labels = labels
        self.learning_rate = lr
        self.hidden_layers = len(hidden_neurons)
        self.wh = [(np.random.randn(hidden_neurons[0], len(dataset[0])) - 0.5).tolist()]
        for i in range(1, self.hidden_layers):
            l = (
                np.random.randn(hidden_neurons[i], hidden_neurons[i - 1]) - 0.5
            ).tolist()
            self.wh.append(l)
        # biases of hidden layers
        # each list represents biases in a layer
        self.bh = [(np.random.randn(hidden_neurons[0]) - 0.5).tolist()]
        for i in range(1, self.hidden_layers):
            self.bh.append((np.random.randn(hidden_neurons[i]) - 0.5).tolist())
        self.wo = (np.random.randn(n_classes, hidden_neurons[-1]) - 0.5).tolist()
        self.bo = (np.random.randn(n_classes) - 0.5).tolist()

    def forward_propagation(self):  # feed forward phase
        self.z = [relu(np.dot(self.x, np.array(self.wh[0]).T) + self.bh[0]).tolist()]
        for i in range(1, self.hidden_layers):
            t = relu(
                np.dot(np.array(self.z[i - 1]), np.array(self.wh[i]).T) + self.bh[i]
            )
            self.z.append(t.tolist())
        self.o = softmax_stable(
            np.dot(np.array(self.z[-1]), np.array(self.wo).T) + self.bo
        )

    def backpropagation(self):  # backpropagation phase
        # softmax derivative
        delta_o = self.o - self.y
        # relu derivative
        delta_h = [
            (relu_derivative(np.array(self.z[-1])) * np.dot(delta_o, self.wo)).tolist()
        ]
        for i in range(self.hidden_layers - 1, 0, -1):
            delta_h.append(
                (
                    relu_derivative(np.array(self.z[i - 1]))
                    * np.dot(delta_h[-1], self.wh[i])
                ).tolist()
            )
        d_bo = delta_o
        self.bo = self.bo - self.learning_rate * d_bo
        delta_h = delta_h[::-1]
        d_bh = delta_h
        for i in range(self.hidden_layers):
            self.bh[i] = self.bh[i] - (self.learning_rate * np.array(d_bh[i]))
        d_wo = np.dot(np.array(delta_o).T, np.array(self.z[-1]))
        self.wo = self.wo - (self.learning_rate * d_wo)

        for i in range(self.hidden_layers - 1, 0, -1):
            d_wh = np.dot(np.array(delta_h[i]).T, np.array(self.z[i - 1]))
            self.wh[i] = self.wh[i] - (self.learning_rate * d_wh)

        d_wh = np.dot(np.array(delta_h[0]).T, np.array(self.x))
        self.wh[0] = self.wh[0] - (self.learning_rate * d_wh)

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
            #     print("prediction:", np.argmax(self.o), "true:", np.argmax(self.y))
        return (count / len(test_dataset)) * 100
