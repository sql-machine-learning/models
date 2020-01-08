import sqlflow_models
from tests.base import BaseTestCases

import tensorflow as tf
import unittest
import numpy as np
from sklearn.datasets import load_iris

class TestDNNClassifier(BaseTestCases.BaseTest):
    def setUp(self):
        x, y = load_iris(return_X_y=True)
        feature_column_names = ['col_{}'.format(d) for d in range(x.shape[1])]
        self.features = {}
        for feature_name, feature_values in zip(feature_column_names, list(x.T)):
            self.features[feature_name] = feature_values
        self.label = y
        feature_columns = [tf.feature_column.numeric_column(key) for key in self.features]
        
        self.model_class = sqlflow_models.DNNClassifier
        self.model = sqlflow_models.DNNClassifier(feature_columns=feature_columns, n_classes=3)

class TestDNNBinaryClassifier(BaseTestCases.BaseTest):
    def setUp(self):
        x, y = load_iris(return_X_y=True)
        x = np.array([x[i] for i, v in enumerate(y) if v != 2])
        y = np.array([y[i] for i, v in enumerate(y) if v != 2])
        feature_column_names = ['col_{}'.format(d) for d in range(x.shape[1])]
        self.features = {}
        for feature_name, feature_values in zip(feature_column_names, list(x.T)):
            self.features[feature_name] = feature_values
        self.label = y
        feature_columns = [tf.feature_column.numeric_column(key) for key in self.features]
        
        self.model_class = sqlflow_models.DNNClassifier
        self.model = sqlflow_models.DNNClassifier(feature_columns=feature_columns, n_classes=2)



if __name__ == '__main__':
    unittest.main()

