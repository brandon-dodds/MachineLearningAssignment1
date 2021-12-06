import csv
import pandas as pd
import numpy as np
import numpy.linalg as linalg


def pol_regression(features_train, y_train, degree):
    pass


def getPolynomialDataMatrix(x, degree):
    X = np.ones(x.shape)
    for i in range(1, degree + 1):
        X = np.column_stack((X, x ** i))
    return X


def getWeightsForPolynomialFit(x, y, degree):
    X = getPolynomialDataMatrix(x, degree)

    XX = X.transpose().dot(X)
    w = np.linalg.solve(XX, X.transpose().dot(y))
    # w = np.linalg.inv(XX).dot(X.transpose().dot(y))

    return w


def main():
    df = pd.read_csv('Task1 - dataset - pol_regression.csv', names=['row_number', 'x', 'y'], skiprows=1)
    x = df.x.to_numpy()
    y = df.y.to_numpy()
    print(getWeightsForPolynomialFit(x, y, 2))


main()
