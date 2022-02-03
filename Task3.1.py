import pandas as pd
from matplotlib import pyplot as plt


def make_plot(series, label):
    series.plot.box()
    plt.ylabel("Alpha")
    plt.xlabel(label)
    plt.show()


def main():
    dataset = pd.read_csv("Task3 - dataset - HIV RVG.csv")
    print(dataset.describe())
    control_set = dataset.loc[dataset['Participant Condition'] == "Control"]
    patient_set = dataset.loc[dataset['Participant Condition'] == "Patient"]
    make_plot(control_set['Alpha'], "Control set")
    make_plot(patient_set['Alpha'], "Patient set")
    plt.xlabel("Beta")
    control_set['Beta'].plot.kde()
    patient_set['Beta'].plot.kde()
    plt.legend(('Control set', 'Patient set'))
    plt.show()


main()
