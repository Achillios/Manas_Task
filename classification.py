from classification_functions import *


if __name__ == "__main__":
    x, y = loadData()
    print('Plotting data with + indicating (y=1) examples and o indicating (y=0) examples\n')
    plotData(x, y)
    input('Press any key to continue')
    m = np.size(x, 0)
    n = np.size(x, 1)
    X = np.c_[np.ones((m, 1))]