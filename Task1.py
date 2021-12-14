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
    y_predicted = generate_one_stack(x, degree).dot(parameters)
    root_mean_square_error = np.sqrt(np.sum(y_predicted, y) ** 2 / len(x))

    return root_mean_square_error


def plot_colour(x, points, y, degree, colour):
    w = pol_regression(x, y, degree)
    x_test = generate_one_stack(points, degree)
    y_test = x_test.dot(w)
    plt.plot(points, y_test, colour)


def main():
    df = pd.read_csv('Task1 - dataset - pol_regression.csv', names=['row_number', 'x', 'y'], skiprows=1)
    x = df.x.to_numpy()
    y = df.y.to_numpy()
    x.sort()
    y.sort()

    generated_x = np.linspace(-5, 5)

    plt.figure()
    plt.plot(x, y, 'bo')

    plot_colour(x, generated_x, y, 0, 'k')
    plot_colour(x, generated_x, y, 1, 'r')
    plot_colour(x, generated_x, y, 2, 'g')
    plot_colour(x, generated_x, y, 3, 'b')
    plot_colour(x, generated_x, y, 6, 'c')
    plot_colour(x, generated_x, y, 10, 'm')

    plt.ylim(-50, 50)
    plt.legend(('training points', '$x^0$', '$x$', '$x^2$', '$x^3$', '$x^6$', '$x^{10}$'))
    plt.show()

    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3)


main()
