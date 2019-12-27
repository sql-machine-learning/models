import sqlflow_models
from tests.base import BaseTestCases

import tensorflow as tf
import numpy as np
import unittest


class TestLSTMBasedTimeSeriesModel(BaseTestCases.BaseTest):
    def setUp(self):
        x = np.array([int(x) for x in range(56)]).reshape(8, 7)
        y = np.array(np.arange(8).reshape(8, 1))
        self.features = {"col1": x}
        self.label = y
        self.n_in = 7
        self.n_out = 1
        # time_window=n_in, num_features=n_out
        feature_columns = [tf.feature_column.numeric_column(key, shape=(self.n_in, self.n_out)) for key in self.features]

        self.model = sqlflow_models.LSTMBasedTimeSeriesModel(
            feature_columns=feature_columns, 
            stack_units=[50, 50], 
            n_in=self.n_in,
            n_out=self.n_out)
        self.model_class = sqlflow_models.LSTMBasedTimeSeriesModel


if __name__ == '__main__':
    unittest.main()

