import pandas as pd
from matplotlib import pyplot as plt


def main():
    dataset = pd.read_csv("Task3 - dataset - HIV RVG.csv")
    print(dataset.describe())


main()
