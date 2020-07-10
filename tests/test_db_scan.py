import sqlflow_models
from tests.base import BaseTestCases
import tensorflow as tf
import unittest
import numpy as np
from sklearn.datasets import load_iris
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn import datasets, metrics
import logging
from pathlib import Path
from numpy import ndarray, testing

iris = datasets.load_iris()
iris_data = np.array(iris.data)
iris_target = iris.target


def purity_score(y_true, y_pred):
    # compute contingency matrix
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def print_in_test(string):
    logging.warning(string)


class TestDBSCAN(unittest.TestCase):
    """DBSCAN test cases."""

    @classmethod
    def setUpClass(self):
        self.dbscan = sqlflow_models.DBSCAN(
            min_samples=10, eps=.4)
        self.dbscan.sqlflow_train_loop(iris_data)

    def test_dbscan_return_labels_with_type_numpy_array(self):
        self.assertIsInstance(self.dbscan.labels_, ndarray)
        print("Test DBSCAN (minpts=10, eps=0.4), the purity score: %f" %
                      purity_score(iris_target, self.dbscan.labels_))


if __name__ == '__main__':
    unittest.main()