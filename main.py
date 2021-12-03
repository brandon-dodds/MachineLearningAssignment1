import csv
import pandas as pd
import numpy as np


def pol_regression(features_train, y_train, degree):
    pass


def getPolynomialDataMatrix(x, degree):
    X = np.ones(x.shape)
    for i in range(1, degree + 1):
        X = np.column_stack((X, x ** i))
    return X


def main():
    df = pd.read_csv('Task1 - dataset - pol_regression.csv', names=['row_number', 'x', 'y'], skiprows=1)
    x = df.x.to_numpy()
    # y = df.y.to_numpy()
    print(x)


main()
