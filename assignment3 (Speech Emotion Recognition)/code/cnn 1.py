from sklearn import datasets
from sklearn.model_selection import train_test_split
from convolution import convolve, max_pooling
import numpy as np
from mlp import MLP
from deep_mlp import DEEP_MLP
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# random seed
np.random.seed(42)

# MNIST DATASET (classes = 10)
dataset = datasets.load_digits()
classes = 10

X_train, X_test, y_train, y_test = train_test_split(
    dataset.data, dataset.target, test_size=0.2, random_state=0
)

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# one-hot encode labels
onehot = OneHotEncoder()
y_train = onehot.fit_transform(y_train.reshape(-1, 1)).toarray()
y_test = onehot.transform(y_test.reshape(-1, 1)).toarray()

# reshape X_train and X_test to 2D images
X_train = X_train.reshape(-1, 8, 8)
# print("X_train shape:", X_train.shape)
X_test = X_test.reshape(-1, 8, 8)

num_filters = 8
filters = np.random.randn(num_filters, 3, 3) / 9


def conv_layer(image, filters):
    num_filters = len(filters)
    conv_result = np.array([])
    for i in range(num_filters):
        t = convolve(image, filters[i], type="2D")
        if conv_result.size == 0:
            conv_result = t
        else:
            conv_result = np.dstack((conv_result, t))
    return conv_result


def max_pooling_layer(conv_result, pool_size):
    num_filters = conv_result.shape[2]
    pooled_result = np.array([])
    for i in range(num_filters):
        t = max_pooling(conv_result[:, :, i], pool_size)
        if pooled_result.size == 0:
            pooled_result = t
        else:
            pooled_result = np.dstack((pooled_result, t))
    return pooled_result


# conv_result = conv_layer(X_train[0], filters)
# pooled_result = max_pooling_layer(conv_result, 2)
result = np.array([])
for i in range(len(X_train)):
    conv_result = conv_layer(X_train[i], filters)
    pooled_result = max_pooling_layer(conv_result, 2)
    # reshape pooled_result to 1D
    pooled_result = pooled_result.reshape(-1, 1)
    # append as a column to result
    if result.size == 0:
        result = pooled_result
    else:
        result = np.hstack((result, pooled_result))

# print("X_train_conv shape:", result.shape)

# print first column of result
# print(result[:, 0])

# hyperparameters
learning_rate = 0.25
maxiter = 15

# for single hidden layer MLP
hidden_neurons = 392
nn = MLP(result.T, y_train, learning_rate, classes, hidden_neurons)
nn.train(maxiter)
print("Accuracy:", nn.predict(result.T, y_train), "%")
# for deep MLP
hidden_neurons_deep = np.array([392])
nn_deep = DEEP_MLP(result.T, y_train, learning_rate, classes, hidden_neurons_deep)
nn_deep.train(maxiter)
print("Accuracy:", nn_deep.predict(result.T, y_train), "%")