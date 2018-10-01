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
    pos = y[:] == 1
    neg = y[:] == 0
    plt.plot(x[pos, 0], x[pos, 1], color='black', marker='+', markersize=7, linewidth=2)
    plt.plot(x[neg, 0], x[neg, 1], color='yellow', marker='o', markersize=7)
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.legend()
    plt.show()
