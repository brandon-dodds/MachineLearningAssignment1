import csv

def pol_regression(features_train, y_train, degree):
    pass


def main():
    with open('Task1 - dataset - pol_regression.csv', 'r') as file:
        reader = csv.reader(file)
        for each_row in reader:
            print(each_row)


main()
