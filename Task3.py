import pandas as pd
from matplotlib import pyplot as plt


def main():
    dataset = pd.read_csv("Task3 - dataset - HIV RVG.csv")
    print(dataset.describe())
    control_set = dataset.loc[dataset['Participant Condition'] == "Control"]
    patient_set = dataset.loc[dataset['Participant Condition'] == "Patient"]
    control_set.boxplot()
    plt.show()


main()
