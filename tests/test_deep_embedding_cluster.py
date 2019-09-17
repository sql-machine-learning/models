import sqlflow_models
from tests.base import BaseTestCases

import tensorflow as tf
import unittest


class TestDeepEmbeddingCluster(BaseTestCases.BaseTest):
    def setUp(self):
        self.features = {"c1": [float(x) for x in range(100)],
                         "c2": [float(x) for x in range(100)],
                         "c3": [float(x) for x in range(100)],
                         "c4": [float(x) for x in range(100)]}
        self.label = [0 for _ in range(50)] + [1 for _ in range(50)]
        feature_columns = [tf.feature_column.numeric_column(key) for key in
                           self.features]
        self.model = sqlflow_models.DNNClassifier(feature_columns=feature_columns)


if __name__ == '__main__':
    unittest.main()

