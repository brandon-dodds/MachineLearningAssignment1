import math
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


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
    centroids = initialise_centroids(dataset, k)
    cluster_assignment = []
    for points in dataset:
        best_centroid = centroids[0]
        for centroid in centroids[1:]:
            if compute_euclidian_distance(points, centroid) < compute_euclidian_distance(points, best_centroid):
                best_centroid = centroid
        cluster_assignment.append((points, best_centroid))

    for centroid in centroids:
        values = []
        mean = 0
        for data in cluster_assignment:
            comparison = centroid == data[1]
            if comparison.all():
                values.append(data[0])

    return centroids, cluster_assignment


def main():
    df = pd.read_csv('Task2 - dataset - dog_breeds.csv',
                     names=['height', 'tail length', 'leg length', 'nose circumference'], skiprows=1)

    x = np.column_stack((df['height'].values, df['tail length'].values))
    y, z = kmeans(x, 3)
    print(y)


main()
