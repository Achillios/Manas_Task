import pandas as pd
import numpy as np


def loadData():
    data = pd.read_csv('ex1data2.txt')
    x = data.iloc[:, 0]
    x = np.c_[x]
    y = data.iloc[:, 1]
    y = np.c_[y]
    m = len(y)
    data.head()
    return x, y


def featureNormalize(x):
    s = x.shape()
    m = s[0]
    n = s[1]
    mu = np.zeros((1, n))
    sigma = np.zeros((1, n))
    mu = np.mean(x)
    sigma = np.std(x)
    xm = np.subtract(x, mu)
    xn = np.divide(xm, sigma)
    return xn, mu, sigma


def gradientMulti(x, y, theta, alpha, num_iters):
    m = len(y)
    J_history = np.zeros((num_iters))
    itt = 1
    while itt != num_iters:
        h = np.dot(x, theta)
        temp = np.dot(np.transpose(x), np.subtract(h, y))
        theta = theta - (alpha / m) * temp
        J_history[itt - 1] = computeCostMulti(x, y, theta)
        num_iters = num_iters + 1
    return theta, J_history


def computeCostMulti(x, y, theta):
    h = np.dot(x, theta)
    m = len(y)
    temp = np.subtract(h, y)
    temp = np.sum(np.power(temp, 2))
    cost = temp / (2 * m)
    return cost
