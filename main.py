import csv
import pandas as pd


def pol_regression(features_train, y_train, degree):
    pass


def main():
    df = pd.read_csv('Task1 - dataset - pol_regression.csv')
    np_array = df.to_numpy()
    print(np_array)


main()
