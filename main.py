import pandas as pd
import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt


def pol_regression(features_train, y_train, degree):
    one_stack = np.ones(features_train.shape)
    for i in range(1, degree + 1):
        one_stack = np.column_stack((one_stack, features_train ** i))

    x_dot_product = one_stack.transpose().dot(one_stack)
    weights = np.linalg.solve(x_dot_product, one_stack.transpose().dot(y_train))

    return weights


def main():
    df = pd.read_csv('Task1 - dataset - pol_regression.csv', names=['row_number', 'x', 'y'], skiprows=1)
    x = df.x.to_numpy()
    y = df.y.to_numpy()
    print(pol_regression(x, y, 2))
    plt.clf()
    plt.scatter(x, y)
    plt.show()


main()
