import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def init_params():
    w1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    w2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return w1, b1, w2, b2


def ReLU(z):
    return np.maximum(z, 0)


def softmax(z):
    x1 = np.exp(z) / sum(np.exp(z))
    return x1


def f_prop(w1, b1, w2, b2, x):
    z1 = w1.dot(x) + b1
    a1 = ReLU(z1)
    z2 = w2.dot(a1) + b2
    a2 = softmax(z2)
    return z1, a1, z2, a2


def ReLU_1(z):
    return z > 0


def one_hot(y):
    one_hot_y = np.zeros((y.size, y.max() + 1))
    one_hot_y[np.arange(y.size), y] = 1
    one_hot_y = one_hot_y.T
    return one_hot_y


def b_prop(z1, a1, z2, a2, w1, w2, x, y):
    one_hot_y = one_hot(y)
    dz2 = a2 - one_hot_y
    dw2 = 1 / m * dz2.dot(a1.T)
    db2 = 1 / m * np.sum(dz2)
    dz1 = w2.T.dot(dz2) * ReLU_1(z1)
    dw1 = 1 / m * dz1.dot(x.T)
    db1 = 1 / m * np.sum(dz1)
    return dw1, db1, dw2, db2


def update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, a):
    w1 = w1 - a * dw1
    b1 = b1 - a * db1
    w2 = w2 - a * dw2
    b2 = b2 - a * db2
    return w1, b1, w2, b2


def get_pred(a2):
    return np.argmax(a2, 0)


def get_acc(predictions, y):
    return np.sum(predictions == y) / y.size


def grad_desc(x, y, a, iter):
    w1, b1, w2, b2 = init_params()

    for i in range(iter):
        z1, a1, z2, a2 = f_prop(w1, b1, w2, b2, x)
        dW1, db1, dW2, db2 = b_prop(z1, a1, z2, a2, w1, w2, x, y)
        w1, b1, w2, b2 = update_params(w1, b1, w2, b2, dW1, db1, dW2, db2, a)

        if i == iter-1:
            predictions = get_pred(a2)
            print('Training Accuracy: ' + str(get_acc(predictions, y)))
    return w1, b1, w2, b2


def make_pred(x, w1, b1, w2, b2):
    _, _, _, a2 = f_prop(w1, b1, w2, b2, x) # f_prop returns z1, a1, z2, a2
    pred = get_pred(a2)
    return pred


def test_pred(index, w1, b1, w2, b2):   
    current_image = x_train[:, index, None]
    pred = make_pred(x_train[:, index, None], w1, b1, w2, b2)
    label = y_train[index]
    print("Prediction: ", pred)
    print("Label: ", label)

    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()


data = pd.read_csv('train.csv')
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

data_dev = data[0:1000].T

y_test = data_dev[0]
x_test = data_dev[1:n]
x_test = x_test / 255

data_train = data[1000:m].T

y_train = data_train[0]
x_train = data_train[1:n]
x_train = x_train / 255
m_train = x_train.shape

w1, b1, w2, b2 = grad_desc(x_train, y_train, 0.10, 500)
dev_predictions = make_pred(x_test, w1, b1, w2, b2)
print('Test Accuracy: ' + str(get_acc(dev_predictions, y_test)))
