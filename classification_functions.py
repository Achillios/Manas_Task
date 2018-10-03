import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def loadData():
    data = pd.read_csv('ex2data1.txt')
    x = data.iloc[:, 0]
    x = np.c_[data.iloc[:, 1], x]
    x = np.c_[x]
    y = data.iloc[:, 2]
    y = np.c_[y]
    data.head()
    return x, y


def plotData(x, y):
    data = pd.read_csv('ex2data1.txt')
    y = data.iloc[:, 2]
    pos = y[:] == 1
    neg = y[:] == 0
    plt.plot([x[pos][:, 0]], [x[pos][:, 1]], color='black', marker='+', markersize=7, linewidth=2)
    plt.plot([x[neg][:, 0]], [x[neg][:, 1]], color='yellow', marker='o', markersize=7)
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')


def sigmoid(z):
    g = np.zeros((np.size(z, 0), np.size(z, 1)))
    temp = 1 + np.exp(-z)
    g = np.divide(1, temp)
    return g


def costFunction(theta, x, y):
    m = len(y)
    n = np.size(x, 1)
    J = 0
    grad = np.zeros((n, 1))
    z = np.dot(x, theta)
    h = sigmoid(z)
    o_y = np.subtract(1, y).T
    s1 = np.dot((-y.T), (np.log(h)))
    s2 = np.dot(o_y, np.log(np.subtract(1, h)))
    s = np.subtract(s1, s2)
    J = (1 / m) * np.sum(s)
    H = sigmoid(x.dot(np.reshape(theta, (-1, 1))))
    grad = (1 / m) * (x.T.dot(H - y))
    return J, grad


def plotDecisionBoundary(theta, x, X, y):
    plotData(x, y)
    plot_x = [[min(X[:, 1]) - 2, max(X[:, 1]) + 2]]
    s1 = np.divide(-1, theta[2])
    s2 = np.multiply(theta[1], plot_x)
    s2 = np.add(theta[0], s2)
    plot_y = np.multiply(s1, s2)
    plot_x = np.c_[plot_x]
    plot_y = np.c_[plot_y]
    plt.plot(plot_x.T, plot_y.T)
    plt.show()


def predict(theta, x):
    z = np.dot(x, theta)
    h = sigmoid(z)
    p = h >= 0.5
    return p.astype('int')
