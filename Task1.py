import pandas as pd
import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def generate_one_stack(x, degree):
    # generates the one stack by creating a stack of ones and column stacking to a specific degree.
    one_stack = np.ones(x.shape)
    for i in range(1, degree + 1):
        one_stack = np.column_stack((one_stack, x ** i))

    return one_stack


def pol_regression(features_train, y_train, degree):
    # Returns the mean if the degree is 0, else it applies the np.linalg.solve the x^2 and y_train.
    if degree == 0:
        return np.mean(y_train)
    one_stack = generate_one_stack(features_train, degree)
    # Transpose and squares it.
    x_dot_product = one_stack.transpose().dot(one_stack)
    # Applies the linalg solve to the values.
    coefficients = np.linalg.solve(x_dot_product, one_stack.transpose().dot(y_train))

    return coefficients


def eval_pol_regression(parameters, x, y, degree):
    y_predicted = generate_one_stack(x, degree).dot(parameters)
    root_mean_square_error = np.sqrt(np.sum((y_predicted - y) ** 2 / len(x)))

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
    x, y = zip(*sorted(zip(x, y)))
    x = np.array(x)
    y = np.array(y)

    generated_x = np.linspace(-5, 5, 200)

    plt.figure()
    plt.plot(x, y, 'bo')

    plot_colour(x, generated_x, y, 0, 'k')
    plot_colour(x, generated_x, y, 1, 'r')
    plot_colour(x, generated_x, y, 2, 'g')
    plot_colour(x, generated_x, y, 3, 'b')
    plot_colour(x, generated_x, y, 6, 'c')
    plot_colour(x, generated_x, y, 10, 'm')
    plt.ylim(-200, 50)
    plt.legend(('training points', '$x^0$', '$x$', '$x^2$', '$x^3$', '$x^6$', '$x^{10}$'))
    plt.show()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    degrees = [0, 1, 2, 3, 6, 10]
    test_rme = []
    train_rme = []
    for degree in degrees:
        train_rme.append(eval_pol_regression(pol_regression(x_train, y_train, degree), x_train, y_train, degree))
        test_rme.append(eval_pol_regression(pol_regression(x_test, y_test, degree), x_test, y_test, degree))

    plt.plot(degrees, test_rme)
    plt.plot(degrees, train_rme)
    plt.legend(('test', 'train'))
    plt.xlabel("Degrees")
    plt.ylabel("RMSE")
    plt.show()


main()
