import math
import pandas as pd
import numpy as np


def compute_euclidian_distance(vec_1, vec_2):
    return math.dist(vec_1, vec_2)


def initialise_centroids(dataset, k):
    centroids = []
    m = len(dataset)
    for i in range(k):
        r = np.random.randint(0, m - 1)
        centroids.append(dataset[r])
    return np.array(centroids)


def kmeans(dataset, k):
    pass


def main():
    df = pd.read_csv('Task2 - dataset - dog_breeds.csv',
                     names=['height', 'tail length', 'leg length', 'nose circumference'], skiprows=1)

    random_centroids = initialise_centroids(df.values, 3)
    print(random_centroids)


main()
