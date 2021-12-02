import csv
import pandas as pd
import numpy as np


def pol_regression(features_train, y_train, degree):
    pass


def main():
    df = pd.read_csv('Task1 - dataset - pol_regression.csv')
    x = df.x.to_numpy()
    y = df.y.to_numpy()
    print(x, y)


main()
