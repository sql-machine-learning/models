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
import pandas as pd
from sklearn import datasets, metrics

def optimizer():
    # SGD is just a placeholder to avoid panic on SQLFLow traning
    return tf.keras.optimizers.SGD(lr=0.1, momentum=0.9)


def loss():
    return None


def prepare_prediction_column(prediction):
    """Return the class label of highest probability."""
    return prediction.argmax(axis=-1)

def purity_score(y_true, y_pred):
    # compute contingency matrix
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


class DBSCAN(tf.keras.Model, BaseEstimator, ClusterMixin):
    OUTLIER = -1

    def __init__(self, min_samples=2, eps=10, feature_columns=None):
        super(DBSCAN, self).__init__(name='DBSCAN')
        self.minpts = min_samples
        self.eps = eps
        self.clusters = []
        self.labels_ = []

    def call(self):
        pass

    def _to_dataframe(self, dataset):
        x_df = pd.DataFrame()
        y_df = pd.DataFrame()

        for features, label in dataset:
            dx = {}
            dy = {}
            for name, value in features.items():
                dx[name] = value.numpy()[0]
            x_df = x_df.append(dx, ignore_index=True)
            if label is not None:
                dy['label'] = label.numpy()[0][0]
                y_df = y_df.append(dy, ignore_index=True)

        if y_df.empty:
            return x_df, None
        return x_df, y_df['label']



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

    def _split_dataset(self, dataset):
        pass

    # do custom training here, parameter "dataset" is a tf.dataset type representing the input data.
    def sqlflow_train_loop(self, dataset, useIrisDemo=True, epochs=1, verbose=0):
        if useIrisDemo == True:
            from sklearn import datasets, metrics
            iris = datasets.load_iris()  # <class 'sklearn.utils.Bunch'>
            x_df = iris.data  # (150, 4) numpy.ndarray float64
            y_df = iris.target
            self.fit_predict(x_df)
            print("DBSCAN (minpts=10, eps=0.4): %f" %
                  purity_score(y_df, self.labels_))
        else:
            x_df, y_df = self._split_dataset(dataset)
            self.fit_predict(x_df)
            print("DBSCAN (minpts=10, eps=0.4): %f" %
                  purity_score(y_df, self.labels_))
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