import sqlflow_models
from tests.base import BaseTestCases

import tensorflow as tf
import unittest
from sklearn.datasets import load_boston


class TestDNNRegressor(BaseTestCases.BaseTest):
    def setUp(self):
        x, y = load_boston(return_X_y=True)
        feature_column_names = ['col_{}'.format(d) for d in range(x.shape[1])]
        self.features = {}
        for feature_name, feature_values in zip(feature_column_names, list(x.T)):
            self.features[feature_name] = feature_values
        self.label = y
        feature_columns = [tf.feature_column.numeric_column(key) for key in self.features]
        self.model_class = sqlflow_models.DNNRegressor
        self.model = sqlflow_models.DNNRegressor(feature_columns=feature_columns)


if __name__ == '__main__':
    unittest.main()

