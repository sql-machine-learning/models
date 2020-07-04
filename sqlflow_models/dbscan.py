#!usr/bin/env python
# -*- coding:utf-8 _*-

"""
__author__ : tiankelang
__email__ : kelang@mail.ustc.edu.cn
__file_name__ : dbscan.py
__create_time__ : 2020/07/01
"""
import numpy as np
import tensorflow as tf
from scipy.spatial.distance import pdist, squareform
from sklearn.base import BaseEstimator, ClusterMixin


class DBSCAN(tf.keras.Model, BaseEstimator, ClusterMixin):
    OUTLIER = -1

    def __init__(self, min_samples=2, eps=10):
        super(DBSCAN, self).__init__()
        self.minpts = min_samples
        self.eps = eps
        self.clusters = []
        self.labels_ = []

    def intersect(self, a, b):
        return len(list(set(a) & set(b))) > 0

    def compute_neighbors(self, distance_matrix):
        neighbors = []
        for i in range(len(distance_matrix)):
            neighbors_under_eps = []
            for neighbor in range(len(distance_matrix[i])):
                if distance_matrix[i][neighbor] <= self.eps \
                        and neighbor != i:
                    neighbors_under_eps.append(neighbor)
            neighbors.append(neighbors_under_eps)
        return neighbors

    def generate_clusters(self, neighbors_list):
        # initiate with the first data
        clusters = [neighbors_list[0] + [0]]
        for i in range(1, len(neighbors_list)):
            # for other data in the neighbour list
            # check if the data has an intersected cluster inside the result list
            # merge the list and append it to the result
            list_of_intersected_cluster = []
            new_cluster = neighbors_list[i] + [i]
            for cluster_num in range(len(clusters)):
                if self.intersect(neighbors_list[i],
                                  clusters[cluster_num]):
                    list_of_intersected_cluster.append(clusters[cluster_num])
                    new_cluster = new_cluster + clusters[cluster_num]

            # if the data is a new cluster / no intersected clusters
            if not list_of_intersected_cluster:
                clusters.append(neighbors_list[i] + [i])
            else:
                clusters.append(list(set(new_cluster)))
                # delete the merged clusters
                for old_cluster in list_of_intersected_cluster:
                    clusters.remove(old_cluster)
        return clusters

    def labelling(self, data, clusters):
        cluster_labels = [self.OUTLIER] * len(data)
        for i in range(len(self.clusters)):
            for j in range(len(self.clusters[i])):
                cluster_labels[self.clusters[i][j]] = i
        return cluster_labels

    def fit(self, X):
        distance_matrix = squareform(pdist(X))
        # compute the neighbors
        neighbors = self.compute_neighbors(distance_matrix)
        # clustering
        self.clusters = self.generate_clusters(neighbors)
        # filter out clusters with neighbors < minpts
        self.clusters = list(filter(lambda x: len(x) >= self.minpts,
                                    self.clusters))
        # labelling
        self.labels_ = np.array(self.labelling(X, self.clusters))

        return self

'''
if __name__ == '__main__':
    from sklearn.datasets.samples_generator import make_blobs
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt
    from sklearn import datasets, metrics

    # iris = datasets.load_iris()
    # iris_data = np.array(iris.data)  # (150, 4) numpy.ndarray float64
    # iris_target = iris.target        # (150,)   numpy.ndarray int64

    centers = [[1, 1], [-1, -1], [1, -1]]
    X, Y = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
                      random_state=0)
    X = StandardScaler().fit_transform(X)

    db = DBSCAN(eps=0.3, min_samples=10)
    label = db.fit(X)
    plt.clf()
    plt.scatter(X[:, 0], X[:, 1], c=db.labels_)
    plt.show()

    # compare with sklearn
    del db
    from sklearn.cluster import DBSCAN
    db = DBSCAN(eps=0.3, min_samples=10)
    db.fit(X)
    plt.clf()
    plt.scatter(X[:, 0], X[:, 1], c=db.labels_)
    plt.show()
    del db
'''