import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def init_params():
    w1 = np.random.rand(128, 784) - 0.5
    b1 = np.random.rand(128, 1) - 0.5
    w2 = np.random.rand(64, 128) - 0.5
    b2 = np.random.rand(64, 1) - 0.5
    w3 = np.random.rand(10, 64) - 0.5
    b3 = np.random.rand(10, 1) - 0.5

    return w1, b1, w2, b2, w3, b3


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)


def tanh(z):
    return np.tanh(z)


def tanh_derivative(z):
    return 1 - np.tanh(z)**2


def relu(z):
    return np.maximum(z, 0)


def relu_derivative(z):
    return z > 0


def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)


def leaky_relu_derivative(z, alpha=0.01):
    return np.where(z > 0, 1, alpha)


def elu(z, alpha=1.0):
    return np.where(z > 0, z, alpha * (np.exp(z) - 1))


def elu_derivative(z, alpha=1.0):
    return np.where(z > 0, 1, alpha * np.exp(z))


def selu(z, __lambda=1.0507, alpha=1.67326):
    return __lambda * np.where(z > 0, z, alpha * (np.exp(z) - 1))


def selu_derivative(z, __lambda=1.0507, alpha=1.67326):
    return __lambda * np.where(z > 0, 1, alpha * np.exp(z))


def softmax(z):
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z, axis=0)


def f_prop(w1, b1, w2, b2, w3, b3, x, activation_function):
    z1 = w1.dot(x) + b1
    a1 = activation_function(z1)
    z2 = w2.dot(a1) + b2
    a2 = activation_function(z2)
    z3 = w3.dot(a2) + b3
    a3 = softmax(z3)

    return z1, a1, z2, a2, z3, a3


def one_hot(y):
    one_hot_y = np.zeros((y.size, y.max() + 1))
    one_hot_y[np.arange(y.size), y] = 1
    one_hot_y = one_hot_y.T

    return one_hot_y


def b_prop(z1, a1, z2, a2, z3, a3, w1, w2, w3, x, y, activation_function_derivative):
    one_hot_y = one_hot(y)
    dz3 = a3 - one_hot_y
    dw3 = 1 / m * dz3.dot(a2.T)
    db3 = 1 / m * np.sum(dz3, axis=1, keepdims=True)
    dz2 = w3.T.dot(dz3) * activation_function_derivative(z2)
    dw2 = 1 / m * dz2.dot(a1.T)
    db2 = 1 / m * np.sum(dz2, axis=1, keepdims=True)
    dz1 = w2.T.dot(dz2) * activation_function_derivative(z1)
    dw1 = 1 / m * dz1.dot(x.T)
    db1 = 1 / m * np.sum(dz1, axis=1, keepdims=True)

    return dw1, db1, dw2, db2, dw3, db3


def update_params(w1, b1, w2, b2, w3, b3, dw1, db1, dw2, db2, dw3, db3, a):
    w1 = w1 - a * dw1
    b1 = b1 - a * db1
    w2 = w2 - a * dw2
    b2 = b2 - a * db2
    w3 = w3 - a * dw3
    b3 = b3 - a * db3

    return w1, b1, w2, b2, w3, b3


def get_pred(a3):
    return np.argmax(a3, 0)


def get_acc(predictions, y):
    return np.sum(predictions == y) / y.size


def grad_desc(x, y, a, iter, activation_function, activation_function_derivative):
    w1, b1, w2, b2, w3, b3 = init_params()

    for j in range(iter):
        z1, a1, z2, a2, z3, a3 = f_prop(w1, b1, w2, b2, w3, b3, x, activation_function)
        dW1, db1, dW2, db2, dW3, db3 = b_prop(z1, a1, z2, a2, z3, a3, w1, w2, w3, x, y, activation_function_derivative)
        w1, b1, w2, b2, w3, b3 = update_params(w1, b1, w2, b2, w3, b3, dW1, db1, dW2, db2, dW3, db3, a)

        if j == iter-1:
            predictions = get_pred(a3)
            print(f'        Training Accuracy for activation function {activation_function}: '
                  f'{get_acc(predictions, y)}')

    return w1, b1, w2, b2, w3, b3


def make_pred(x, w1, b1, w2, b2, w3, b3, activation_function):
    _, _, _, _, _, a3 = f_prop(w1, b1, w2, b2, w3, b3, x, activation_function)
    return get_pred(a3)


def test_pred(index, w1, b1, w2, b2, w3, b3, activation_function):
    current_image = x_train[:, index, None]
    pred = make_pred(x_train[:, index, None], w1, b1, w2, b2, w3, b3, activation_function)
    label = y_train[index]
    print("Prediction: ", pred)
    print("Label: ", label)

    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()


def evaluate_model(x, y, w1, b1, w2, b2, w3, b3, activation_function):
    predictions = make_pred(x, w1, b1, w2, b2, w3, b3, activation_function)
    acc = get_acc(predictions, y)
    print('Test Accuracy:', acc)

    return acc


activation_functions = {
    'sigmoid': (sigmoid, sigmoid_derivative),
    'tanh': (tanh, tanh_derivative),
    'relu': (relu, relu_derivative),
    'leaky_relu': (leaky_relu, leaky_relu_derivative),
    'elu': (elu, elu_derivative),
    'selu': (selu, selu_derivative)
}

data = pd.read_csv('train.csv')
data = np.array(data)
print(f"Shape of input data: {data.shape}")
m, n = data.shape
np.random.shuffle(data)

data_dev = data[0:100].T

y_test = data_dev[0]
x_test = data_dev[1:n]
x_test = x_test / 255.0
# x_test = (x_test / 127.5) - 1

data_train = data[100:m].T

y_train = data_train[0]
x_train = data_train[1:n]
x_train = x_train / 255.0
# x_train = (x_train / 127.5) - 1
m_train = x_train.shape

performance_index = {}

for name, (activation_function, activation_function_derivative) in activation_functions.items():
    print(f"Training for {name.capitalize()} activation function")
    w1, b1, w2, b2, w3, b3 = grad_desc(x_train, y_train, 0.01, 1000,
                                       activation_function, activation_function_derivative)
    performance_index[name] = evaluate_model(x_test, y_test, w1, b1, w2, b2, w3, b3, activation_function)

sorted_performance_index = sorted(performance_index.items(), key=lambda item: item[1], reverse=True)
print(f"Order of performance of activation functions is {sorted_performance_index}")