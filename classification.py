from classification_functions import *


if __name__ == "__main__":
    x, y = loadData()
    print('Plotting data with + indicating (y=1) examples and o indicating (y=0) examples\n')
    plotData(x, y)
    plt.show()
    input('Press any key to continue')
    m = np.size(x, 0)
    n = np.size(x, 1)
    X = np.c_[np.ones((m, 1)), x]
    initial_theta = np.zeros((n+1, 1))
    cost, grad = costFunction(initial_theta, X, y)
    print('Cost at initial theta(zeros):\n', cost)
    print('Expected cost (approx): 0.693\n')
    print('Gradient at initial theta(zeros):\n', grad)
    print('Expected gradient(approx):\n-0.1000\n-12.0092\n-11.2628')
    # compute and display cost and gradient with non-zero theta
    test_theta = [[-24], [0.2], [0.2]]
    cost, grad = costFunction(test_theta, X, y)
    print('Cost at test theta:\n', cost)
    print('Expected cost (approx): 0.218\n')
    print('Gradient at test theta:\n', grad)
    print('Expected Gradient value (approx):\n0.043\n2.566\n2.647\n')
    theta = [[-25.161], [0.206], [0.201]]
    input('Press any key to continue')
    # plot decision boundary
    plotDecisionBoundary(theta, x, X, y)
    input('Press any key to continue')
    # predict part
    temp = [[1, 45, 85]]
    temp = np.dot(temp, theta)
    prob = sigmoid(temp)
    print('For a student with scores 45 and 85, we predict an admission probability of ', prob)
    print('Expected value: 0.775 +/- 0.002\n')
