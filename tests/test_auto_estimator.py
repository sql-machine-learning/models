import sqlflow_models
from tests.base import BaseTestCases, train_input_fn, eval_input_fn

import sys
import tensorflow as tf
import unittest
import numpy as np
from sklearn.datasets import load_iris, load_boston

class TestAutoClassifier(BaseTestCases.BaseEstimatorTest):
    def setUp(self):
        x, y = load_iris(return_X_y=True)
        feature_column_names = ['col_{}'.format(d) for d in range(x.shape[1])]
        self.features = {}
        for feature_name, feature_values in zip(feature_column_names, list(x.T)):
            self.features[feature_name] = feature_values
        self.label = y
        feature_columns = [tf.feature_column.numeric_column(key) for key in self.features]
        
        self.model_class = sqlflow_models.AutoClassifier
        self.model = sqlflow_models.AutoClassifier(feature_columns=feature_columns, n_classes=3)

class TestAutoBinaryClassifier(BaseTestCases.BaseEstimatorTest):
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
        
        self.model_class = sqlflow_models.AutoClassifier
        self.model = sqlflow_models.AutoClassifier(feature_columns=feature_columns)

class TestAutoRegressor(BaseTestCases.BaseEstimatorTest):
    def setUp(self):
        x, y = load_boston(return_X_y=True)
        feature_column_names = ['col_{}'.format(d) for d in range(x.shape[1])]
        self.features = {}
        for feature_name, feature_values in zip(feature_column_names, list(x.T)):
            self.features[feature_name] = feature_values
        self.label = y
        feature_columns = [tf.feature_column.numeric_column(key) for key in self.features]
        self.model_class = sqlflow_models.AutoRegressor
        self.model = sqlflow_models.AutoRegressor(feature_columns=feature_columns)

if __name__ == '__main__':
    unittest.main()

