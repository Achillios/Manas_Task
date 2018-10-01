from functions_multi import *

if __name__ == "__main__":
    x, y = loadData()
    m = len(y)
    input('Press any key to continue')
    print('Normalizing Features...\n')
    x, mu, sigma = featureNormalize(x)
    X = np.c_[np.ones((m, 1)), x]
    print('Running gradient Descent')
    theta = np.zeros(3, 1)
    alpha = 0.01
    num_iters = 400
    theta, J_history = gradientMulti(X, y, theta, alpha, num_iters)
