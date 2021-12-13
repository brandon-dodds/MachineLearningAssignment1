import pandas as pd
import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def generate_one_stack(x, degree):
    one_stack = np.ones(x.shape)
    for i in range(1, degree + 1):
        one_stack = np.column_stack((one_stack, x ** i))

    return one_stack


def pol_regression(features_train, y_train, degree):
    if degree == 0:
        return np.mean(y_train)
    one_stack = generate_one_stack(features_train, degree)
    x_dot_product = one_stack.transpose().dot(one_stack)
    weights = np.linalg.solve(x_dot_product, one_stack.transpose().dot(y_train))

    return weights


def eval_pol_regression(parameters, x, y, degree):
    pass
    # rmse = np.sqrt(np.sum(y_predicted, y_actual) ** 2 / len(x))


def main():
    df = pd.read_csv('Task1 - dataset - pol_regression.csv', names=['row_number', 'x', 'y'], skiprows=1)
    x = df.x.to_numpy()
    y = df.y.to_numpy()
    x.sort()
    y.sort()

    x_test = np.linspace(-5, 5, 50)

    plt.figure()
    plt.plot(x, y, 'bo')

    w0 = pol_regression(x, y, 0)
    x_test_0 = generate_one_stack(x_test, 0)
    y_test_0 = x_test_0.dot(w0)
    plt.plot(x_test, y_test_0, 'k')

    w1 = pol_regression(x, y, 1)
    x_test_1 = generate_one_stack(x_test, 1)
    y_test_1 = x_test_1.dot(w1)
    plt.plot(x_test, y_test_1, 'r')

    w2 = pol_regression(x, y, 2)
    x_test_2 = generate_one_stack(x_test, 2)
    y_test_2 = x_test_2.dot(w2)
    plt.plot(x_test, y_test_2, 'g')

    w3 = pol_regression(x, y, 3)
    x_test_3 = generate_one_stack(x_test, 3)
    y_test_3 = x_test_3.dot(w3)
    plt.plot(x_test, y_test_3, 'b')

    w4 = pol_regression(x, y, 6)
    x_test_4 = generate_one_stack(x_test, 6)
    y_test_4 = x_test_4.dot(w4)
    plt.plot(x_test, y_test_4, 'c')

    w5 = pol_regression(x, y, 10)
    x_test_5 = generate_one_stack(x_test, 10)
    y_test_5 = x_test_5.dot(w5)
    plt.plot(x_test, y_test_5, 'm')

    plt.ylim(-50, 50)
    plt.legend(('training points', '$x^0$', '$x$', '$x^2$', '$x^3$', '$x^6$', '$x^{10}$'))
    plt.show()


main()
