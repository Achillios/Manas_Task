from functions import *

if __name__ == "__main__":
    print('Running warmupexercise...\n')
    print('5x5 Identity Matrix:\n')
    warmupexercise()
    input('Press any key to continue')
    print('Plotting Data...\n')
    data = pd.read_csv('ex1data1.txt', header=None)
    x = data.iloc[:, 0]
    y = data.iloc[:, 1]
    m = len(y)
    data.head()
    plotData(x, y)
    input('Press any key to continue')
    X = np.c_[np.ones((m, 1)), x]
    theta = np.zeros((2, 1))
    iterations = 1500
    alpha = 0.01
    print('\nTesting the cost function...\n')
    J = computeCost(X, y, theta)
    print('With theta = [0 ; 0]\nCost computed = ', J)
    print('Expected cost value (approx) 32.07\n')
    J = computeCost(X, y, [[-1], [2]])
    print('With theta = [-1 ; 2]\n Cost computed = ', J)
    print('Expected cost value (approx) 54.24\n')
    input('Press any key to continue')
    theta = gradientDescent(X, y, theta, alpha, iterations)
    print('Theta found by gradient descent:\n')
    print(theta)
    print('Expected theta values (approx)\n')
    print('-3.603\n1.1664\n')
    plt.plot([x], [y], 'rx', MarkerSize=7)
    plt.plot(X[:, 1], X.dot(theta), '-')
    # plt.legend('Training data', 'Linear regression')
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')
    plt.axis([4, 24, -5, 25])
    plt.show()
    predict1 = [1, 3.5]
    predict1 = np.dot(predict1, theta)
    print('For population = 35,000, we predict profit of ', (predict1*10000))
    predict2 = [1, 7]
    predict2 = np.dot(predict2, theta)
    print('For population = 70,000, we predict a profit of ', (predict2*10000))
    input("Press any key ro continue")
    print('Visualizing J(theta_0, theta_1)...\n')
    theta0_vals = np.linspace(-10, 10, 100)
    theta1_vals = np.linspace(-1, 4, 100)
    J_vals = np.zeros((len(theta0_vals.T), len(theta1_vals.T)))
    p =0
    q = 0
    while p != 100:
        while q != 100:
            t = [[theta0_vals[p]], [theta1_vals[q]]]
            J_vals[p, q] = computeCost(X, y, t)
            q = q + 1
        p = p + 1
    Axes3D.plot_surface(theta0_vals, theta1_vals, np.transpose(J_vals))
    Axes3D.set_xlabel('theta_0')
    Axes3D.set_ylabel('theta_1')



