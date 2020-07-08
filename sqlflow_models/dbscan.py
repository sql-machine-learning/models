#!usr/bin/env python
# -*- coding:utf-8 _*-

"""
__author__ : tiankelang
__email__ : kelang@mail.ustc.edu.cn
__file_name__ : dbscan.py
__create_time__ : 2020/07/01

demo iris:
%%sqlflow
SELECT * FROM iris.train
TO TRAIN sqlflow_models.DBSCAN
WITH
model.min_samples=10,
model.eps=0.3
INTO sqlflow_models.my_dbscan_model;
"""
import tensorflow as tf
from scipy.spatial.distance import pdist, squareform
from sklearn.base import BaseEstimator, ClusterMixin
import pandas as pd
from sklearn import datasets, metrics
import numpy as np
from scipy.spatial import KDTree
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
import six

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


class DBSCAN(tf.keras.Model):
    def __init__(self,
                 eps: float = 0.5,
                 min_samples: int = 5,
                 has_label=False,
                 feature_columns=None):
        '''
        :param eps: Neighborhood distance
        :param min_samples:
        The minimum number of samples required to form a class cluster
        '''
        super(DBSCAN, self).__init__(name='DBSCAN')
        self.eps = eps
        self.min_samples = min_samples
        self.core_sample_indices_ = list()
        self.components_ = None
        self.labels_ = None
        self.has_label = has_label

    def fit_predict(self, X):
        n_samples = len(X)

        kd_tree = KDTree(X)  # build KDTree

        density_arr = np.array([len(kd_tree.query_ball_point(x, self.eps)) for x in X])  # 密度数组

        visited_arr = [False for _ in range(n_samples)]  # Access tag array

        k = -1  # init class
        self.labels_ = np.array([-1 for _ in range(n_samples)])

        for sample_idx in range(n_samples):
            if visited_arr[sample_idx]:  # Skip visited samples
                continue

            visited_arr[sample_idx] = True

            # Skip noise samples and boundary samples
            if density_arr[sample_idx] == 1 or density_arr[sample_idx] < self.min_samples:
                continue

            # core object
            else:
                # Find all the core objects in the neighborhood, including themselves
                cores = [idx for idx in kd_tree.query_ball_point(X[sample_idx], self.eps) if
                         density_arr[idx] >= self.min_samples]
                k += 1
                self.labels_[sample_idx] = k
                self.core_sample_indices_.append(sample_idx)

                while cores:
                    cur_core = cores.pop(0)
                    if not visited_arr[cur_core]:
                        self.core_sample_indices_.append(cur_core)
                        visited_arr[cur_core] = True
                        self.labels_[cur_core] = k

                        neighbors = kd_tree.query_ball_point(X[cur_core], self.eps)
                        neighbor_cores = [idx for idx in neighbors if
                                          idx not in cores and density_arr[idx] >= self.min_samples]
                        neighbor_boards = [idx for idx in neighbors if density_arr[idx] < self.min_samples]

                        cores.extend(neighbor_cores)

                        for idx in neighbor_boards:
                            if self.labels_[idx] == -1:
                                self.labels_[idx] = k

        # Update class properties
        self.core_sample_indices_ = np.sort(np.array(self.core_sample_indices_))
        self.components_ = X[self.core_sample_indices_.astype('int64')]
        return self.labels_


    def _read_Dataset_data(self, dataset):
        data = None
        label = None
        flag = True
        print("dataset:", dataset)
        for item in dataset:
            # print("item:", item)
            if flag:
                flag = False
                item_data = item[0]     # dict
                len1 = len(item_data)
                index=0

                feature_data = []
                feature_column_names = []

                for k, v in item_data.items():
                    if index == (len1-1):
                        item_label = v.numpy().reshape(1, )
                    else:
                        feature_column_names.append(k)
                        feature_data.append(v.numpy())
                        index = index + 1
                feature_data = np.asarray(feature_data).reshape(1, -1)

                data = np.asarray(feature_data).reshape(1, -1)
                label = item_label
            else:
                item_data = item[0]
                len1 = len(item_data)
                index = 0

                feature_data = []
                feature_column_names = []

                for k, v in item_data.items():
                    if index == (len1 - 1):
                        item_label = v.numpy().reshape(1, )
                    else:
                        feature_column_names.append(k)
                        feature_data.append(v.numpy())
                        index = index + 1
                feature_data = np.asarray(feature_data).reshape(1, -1)

                data = np.concatenate((data, feature_data), axis=0)
                label = np.concatenate((label, item_label), axis=0)
        print("data:", type(data), data.shape)
        print("label:", type(label), label.shape)
        return data, label
    # do custom training here, parameter "dataset" is a tf.dataset type representing the input data.
    def sqlflow_train_loop(self, dataset, epochs=1, verbose=0):
        '''
        Parameter `epochs` and `verbose` will not be used in this function. :param dataset: demo iris,
        :param dataset:
        demo iris <class 'tensorflow.python.data.ops.dataset_ops.DatasetV1Adapter'>
        <DatasetV1Adapter shapes: ({sepal_length: (1,), sepal_width: (1,), petal_length: (1,), petal_width: (1,)},
        (1, None)), types: ({sepal_length: tf.float32, sepal_width: tf.float32, petal_length: tf.float32,
        petal_width: tf.float32}, tf.int64)>
        :param epochs:
        :param verbose:
        :return:
        '''
        data, label = self._read_Dataset_data(dataset)

        self.fit_predict(data)
        print("DBSCAN(eps= %.2f, minpts= %d), the purity score: %f" %
              (self.eps,
               self.min_samples,
               purity_score(label, self.labels_)))
        # print("Predict labels:", self.labels_)
        # print("True labels:", label)
