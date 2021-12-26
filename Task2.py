import math
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def plot_scatter_curve(k_means):
    centroids = k_means[0]
    cluster_assignment = k_means[1]

    for x in centroids:
        centroid_list = []
        for y in cluster_assignment:
            if np.array_equal(y[1], x):
                centroid_list.append(y[0])
        centroid_list = np.array(centroid_list)
        plt.scatter(centroid_list[:, 0], centroid_list[:, 1])
    centroids = np.array(centroids)
    plt.scatter(centroids[:, 0], centroids[:, 1], color='black')
    plt.show()


def compute_euclidian_distance(vec_1, vec_2):
    return math.dist(vec_1, vec_2)


def initialise_centroids(dataset, k):
    centroids = []
    m = len(dataset)
    for i in range(k):
        r = np.random.randint(0, m - 1)
        centroids.append(dataset[r])
    return centroids


def kmeans(dataset, k):
    cluster_assignment = []
    centroids = initialise_centroids(dataset, k)
    mean_changed = True
    while mean_changed:
        cluster_assignment = []
        new_centroids = []
        for points in dataset:
            best_centroid = centroids[0]
            for centroid in centroids[1:]:
                if compute_euclidian_distance(points, centroid) < compute_euclidian_distance(points, best_centroid):
                    best_centroid = centroid
            cluster_assignment.append((points, best_centroid))

        for centroid in centroids:
            values = []
            for data in cluster_assignment:
                if np.array_equal(centroid, data[1]):
                    values.append(data[0])
            mean = np.mean(values, axis=0)
            new_centroids.append(mean)
        if np.array_equal(centroids, new_centroids):
            break
        else:
            centroids = new_centroids

    return centroids, cluster_assignment


def main():
    df = pd.read_csv('Task2 - dataset - dog_breeds.csv',
                     names=['height', 'tail length', 'leg length', 'nose circumference'], skiprows=1)

    height_tail_length = np.column_stack((df['height'].values, df['tail length'].values))
    height_leg_length = np.column_stack((df['height'].values, df['leg length'].values))
    plot_scatter_curve(kmeans(height_tail_length, 2))
    plot_scatter_curve(kmeans(height_tail_length, 3))
    plot_scatter_curve(kmeans(height_leg_length, 2))
    plot_scatter_curve(kmeans(height_leg_length, 3))


main()
