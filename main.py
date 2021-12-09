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
    degree = 10

    plt.figure()
    plt.plot(x, y, 'bo')
    w1 = pol_regression(x, y, degree)
    Xtest1 = generate_one_stack(x, degree)
    ytest1 = Xtest1.dot(w1)
    plt.plot(x, ytest1, 'r')
    plt.show()


main()
