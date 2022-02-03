import pandas as pd
from matplotlib import pyplot as plt


def main():
    dataset = pd.read_csv("Task3 - dataset - HIV RVG.csv")
    print(dataset.describe())
    control_set = dataset.loc[dataset['Participant Condition'] == "Control"]
    patient_set = dataset.loc[dataset['Participant Condition'] == "Patient"]
    control_set['Alpha'].plot.box()
    plt.xlabel("Control set")
    plt.show()
    patient_set['Alpha'].plot.box()
    plt.xlabel("Patient set")
    plt.show()
    control_set['Beta'].plot.kde()
    patient_set['Beta'].plot.kde()
    plt.show()


main()
