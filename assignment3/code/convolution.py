import numpy as np
from math import ceil, floor

# 1d convolution with padding and stride -> done
# 2d convolution with padding and stride -> done
# 3d convolution with padding and stride -> need testing


# 1D convolution: generate a 1D vector
def convolve_1D(x, w, stride):
    # x is the input, w is the kernel
    # x is n x 1
    # w is k x 1
    # k <= n (k is window size)
    n = len(x)
    k = len(w)
    # result size is (t+1)
    t = floor((n - k) / stride)
    result = []
    for i in range(t + 1):
        # using numpy dot product
        result.append(np.dot(x[i * stride : i * stride + k], w))
    return np.array(result)  # (t+1)


# 2D convolution: generate a 2D matrix
def convolve_2D(x, w, stride):
    # x is the input, w is the kernel
    # x is n x n
    # w is k x k
    # k <= n (k is window size)
    n, n = x.shape
    k, k = w.shape
    # result size is (t+1)x(t+1)
    t = floor((n - k) / stride)
    result = []
    for i in range(t + 1):
        result.append([])
        for j in range(t + 1):
            # using numpy dot product
            result[i].append(
                np.dot(
                    x[
                        i * stride : i * stride + k, j * stride : j * stride + k
                    ].flatten(),
                    w.flatten(),
                )
            )
    return np.array(result)  # (t+1) x (t+1)


# 3D convolution: generate a 2D matrix
def convolve_3D(x, w, stride):
    # a 3D matrix is called a tensor
    # first dimension comprises the rows
    # second dimension comprises the columns
    # third dimension comprises the channels
    # x is the input, w is the kernel
    # x is n x n x m tensor (m channels)
    # x is a collection of n x n matrices obtained by applying m filters
    # example:
    # image of n x n pixels with m channels (RGB)
    # channel 1: red
    # channel 2: green
    # channel 3: blue
    # w is k x k x r
    # k <= n (k is window size)
    # r <= m (r is number of filters)
    # we always assume that the number of channels in the input is equal to the number of filters
    # w has only one parameter which is the window size
    # for each channel apply its corresponding filter
    # result is a 2d matrix of size (n-k+1) x (n-k+1)
    # result is sum of the results of each channel
    # for each channel apply its corresponding filter
    m = len(x)
    result = []
    for i in range(m):
        result.append(convolve_2D(x[i], w[i], stride))
    # sum of the results of each channel
    return np.sum(result, axis=0)


def convolve(x, w, type="3D", padding=False, stride=1):
    p = ceil((len(w) - 1) / 2)
    if type == "1D":
        if padding:
            x = np.pad(x, (p,), "constant")
        return convolve_1D(x, w, stride)
    elif type == "2D":
        if padding:
            x = np.pad(x, ((p, p), (p, p)), "constant")
        return convolve_2D(x, w, stride)
    elif type == "3D":
        # add padding to each channel
        if padding:
            x = np.pad(x, ((0, 0), (p, p), (p, p)), "constant")
        return convolve_3D(x, w, stride)


# max-pooling uses identity activation function
def max_pooling(x, p):
    n, n = x.shape
    w = np.ones((p, p))
    # result size
    stride = p
    s = floor(n / stride)
    result = []
    # replace dot product with max
    for i in range(s):
        result.append([])
        for j in range(s):
            result[i].append(
                np.max(x[i * stride : i * stride + p, j * stride : j * stride + p])
            )
    return np.array(result)

# max-pooling on a 1D vector 
def max_pooling_1D(x, p):
    n = len(x)
    # result size
    stride = p
    s = floor(n / stride)
    result = []
    # replace dot product with max
    for i in range(s):
        result.append(
            np.max(x[i * stride : i * stride + p])
        )
    return np.array(result)


def avg_pooling(x, p):
    n, n = x.shape
    w = np.ones((p, p))
    # result size
    stride = p
    s = floor(n / stride)
    result = []
    # replace dot product with mean
    for i in range(s):
        result.append([])
        for j in range(s):
            result[i].append(
                np.mean(x[i * stride : i * stride + p, j * stride : j * stride + p])
            )
    return np.array(result)


def max_unpooling(x, m, p):
    n, n = x.shape
    stride = p
    s = floor(n / stride)
    result = np.zeros((n, n))
    for i in range(s):
        for j in range(s):
            # find the index of the maximum element in the corresponding
            # submatrix of x
            index = np.argmax(
                x[i * stride : i * stride + p, j * stride : j * stride + p]
            )
            # find the row and column of the maximum element
            row = index // p
            col = index % p
            # place the element in the correct position in x
            result[i * stride + row, j * stride + col] = m[i, j]
    return result


def transposed_conv(x, m):
    n, n = x.shape
    k, k = m.shape
    size = n + k - 1
    result = np.zeros((size, size))
    for i in range(n):
        for j in range(n):
            result[i : i + k, j : j + k] += x[i, j] * m
    return result


x_1D = np.array([1, 3, -1, 2, 3, 1, -2])
w_1D = np.array([1, 0, 2])

x_2D = np.array(
    [
        [1, 2, 2, 1],
        [3, 1, 4, 2],
        [2, 1, 3, 4],
        [1, 2, 3, 1],
    ]
)
w_2D = np.array(
    [
        [1, 0],
        [0, 1],
    ]
)

x_3D = np.array(
    [
        [
            [1, -2, 4],
            [2, 1, -2],
            [1, 3, -1],
        ],
        [
            [2, 1, 3],
            [3, -1, 1],
            [1, 1, -2],
        ],
        [
            [1, -1, 3],
            [2, 1, 4],
            [3, 1, 2],
        ],
    ]
)
w_3D = np.array(
    [
        [[0, 1], [1, 0]],
        [[1, 0], [0, 1]],
        [[1, 1], [2, 0]],
    ]
)
