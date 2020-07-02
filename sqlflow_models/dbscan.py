#!usr/bin/env python
# -*- coding:utf-8 _*-

"""
__author__ : tiankelang
__email__ : kelang@mail.ustc.edu.cn
__file_name__ : dbscan.py
__create_time__ : 2020/07/01
"""
import numpy as np

class DBSCAN:
    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        '''
        :param eps: Neighborhood distance
        :param min_samples:
            The minimum number of samples required to form a cluster
        '''
        self.eps = eps
        self.min_samples = min_samples
        self.core_sample_indices_ = list()
        self.components_ = None
        self.labels_ = None

    def euclidean_distances(self, X, Y=None, Y_norm_squared=None, X_norm_squared=None):
        '''
        Each row of data is regarded as a sample,
        and the Euclidean distance between two matrix samples is calculated
        :param X: matrix one
        :param Y: matrix two
        :param Y_norm_squared:
        :param X_norm_squared:
        :return: pairwise  Distance matrix
        '''
        X = np.array(X)
        Y = np.array(Y) if Y else X

        dist_mat = np.dot(X, Y.T)

        X_squared = np.sum(np.square(X), axis=1).reshape((dist_mat.shape[0], -1))
        Y_squared = np.sum(np.square(Y), axis=1).reshape((-1, dist_mat.shape[1]))
        squared_dist = X_squared - 2 * dist_mat + Y_squared
        squared_dist[squared_dist < 0] = 0
        # Negative numbers may appear under some data, so it needs to be truncated

        return np.sqrt(squared_dist)

    def fit(self, X):
        dist_mat = self.euclidean_distances(X)
        dens_arr = list()  # Density array
        for row in dist_mat:
            dens = np.sum(row <= self.eps)  # Density calculate
            dens_arr.append(dens)
        dens_arr = np.array(dens_arr)
        visited_arr = [False for _ in range(len(X))]  # Access tag array
        self.labels_ = [-1 for _ in range(len(X))]  # Category
        k = -1  # all sample points are noise points by default

        # Traversing sample points
        for idx in range(len(X)):
            if visited_arr[idx]:  # If it has been accessed, it will be skipped
                continue

            visited_arr[idx] = True

            if dens_arr[idx] == 1 or dens_arr[idx] < self.min_samples:  # Noise sample or boundary
                continue

            else:  # core object
                # Access the queue, which is modified in the loop
                cores_q = [i for i in range(
                    len(X)) if dist_mat[i, idx] <= self.eps and dens_arr[i] >= self.min_samples]
                k += 1  # New category
                self.labels_[idx] = k  # Assign a category to the current core object

                while cores_q:  # BFS  Density linked core objects
                    cur_core = cores_q.pop(0)

                    # For the core object that has not been accessed,
                    # the core that has been accessed will be skipped directly
                    if not visited_arr[cur_core]:
                        visited_arr[cur_core] = True
                        self.labels_[cur_core] = k

                        neighbors = [i for i in range(
                            len(X)) if dist_mat[i, cur_core] <= self.eps]
                        neighbor_cores = [
                            i for i in neighbors if i not in cores_q and dens_arr[i] >= self.min_samples]  # core objects in the neighborhood
                        neighbor_boards = [
                            i for i in neighbors if dens_arr[i] < self.min_samples]  # boundary samples in the neighborhood

                        # core points join the queue to wait for access
                        cores_q.extend(neighbor_cores)

                        # boundary points are classified
                        for node_idx in neighbor_boards:
                            if self.labels_[node_idx] == -1:
                                self.labels_[node_idx] = k

    def call(self, inputs):
        self.fit(inputs)
        return self.labels_

    def sqlflow_train_loop(self, x, epochs=1, verbose=0):
        pass

    def display_model_info(self, verbose=0):
        pass

def loss(*args, **kwargs):
    return None

def optimizer(*args, **kwargs):
    return None


# if __name__ == '__main__':
#     from sklearn.datasets.samples_generator import make_blobs
#     from sklearn.preprocessing import StandardScaler
#
#     centers = [[1, 1], [-1, -1], [1, -1]]
#     X, Y = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
#                       random_state=0)
#     X = StandardScaler().fit_transform(X)
#
#     db = DBSCAN(eps=0.3, min_samples=10)
#     db.call(X)
#
#     import matplotlib.pyplot as plt
#
#     plt.clf()
#     plt.scatter(X[:, 0], X[:, 1], c=db.labels_)
#     plt.show()
#
#     # 对比sklearn
#     del db
#     from sklearn.cluster import DBSCAN
#
#     db = DBSCAN(eps=0.3, min_samples=10)
#     db.fit(X)
#     plt.clf()
#     plt.scatter(X[:, 0], X[:, 1], c=db.labels_)
#     plt.show()