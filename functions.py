import numpy as np
import pandas as pd
import matplotlib as mp
import matplotlib.pyplot as plt


def warmupexercise():
    i = np.identity(5, int)
    print(i)


def plotData(x, y):
    plt.plot([x], [y], 'rx', MarkerSize=7)
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')
    plt.axis([4, 24, -5, 25])
    plt.show()


def computeCost(x, y, theta):
    m = len(y)
    j = 0
    h = x.dot(theta)
    temp = h - np.transpose([y])
    temp = np.power(temp, 2)
    j = (1 / (2 * m)) * temp.sum()
    return j


def gradientDescent(x, y, theta, alpha, iteration):
    J = np.zeros(iteration)
    m = len(y)
    y = np.c_[y]
    while iteration != 0:
        h = np.dot(x, theta)
        temp = np.subtract(h, y)
        ins = np.dot(x[:, 0], temp)
        theta[0] = theta[0] - alpha*ins/m
        ins = np.dot(x[:, 1], temp)
        theta[1] = theta[1] - alpha*ins/m
        iteration = iteration - 1
    return theta
