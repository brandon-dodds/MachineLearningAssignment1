import pandas as pd
import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt


def generate_one_stack(x, degree):
    one_stack = np.ones(x.shape)
    for i in range(1, degree + 1):
        one_stack = np.column_stack((one_stack, x ** i))

    return one_stack


def pol_regression(features_train, y_train, degree):
    one_stack = generate_one_stack(features_train, degree)
    x_dot_product = one_stack.transpose().dot(one_stack)
    weights = np.linalg.solve(x_dot_product, one_stack.transpose().dot(y_train))

    return weights


def main():
    df = pd.read_csv('Task1 - dataset - pol_regression.csv', names=['row_number', 'x', 'y'], skiprows=1)
    x = df.x.to_numpy()
    y = df.y.to_numpy()
    x.sort()
    y.sort()

    plt.figure()
    plt.plot(x, y, 'bo')

    w1 = pol_regression(x, y, 1)
    Xtest1 = generate_one_stack(x, 1)
    ytest1 = Xtest1.dot(w1)
    plt.plot(x, ytest1, 'r')

    w2 = pol_regression(x, y, 2)
    Xtest2 = generate_one_stack(x, 2)
    ytest2 = Xtest2.dot(w2)
    plt.plot(x, ytest2, 'g')

    w3 = pol_regression(x, y, 3)
    Xtest3 = generate_one_stack(x, 3)
    ytest3 = Xtest3.dot(w3)
    plt.plot(x, ytest3, 'b')

    w4 = pol_regression(x, y, 6)
    Xtest4 = generate_one_stack(x, 6)
    ytest4 = Xtest4.dot(w4)
    plt.plot(x, ytest4, 'c')

    w5 = pol_regression(x, y, 10)
    Xtest5 = generate_one_stack(x, 10)
    ytest5 = Xtest5.dot(w5)
    plt.plot(x, ytest5, 'm')

    plt.legend(('training points', '$x$', '$x^2$', '$x^3$', '$x^6$', '$x^{10}$'))
    plt.show()


main()
