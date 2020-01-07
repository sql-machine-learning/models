import sqlflow_models
from tests.base import BaseTestCases

import tensorflow as tf
import numpy as np
import unittest


class TestStackedBiLSTMClassifier(BaseTestCases.BaseTest):
    def setUp(self):
        self.features = {"c1": np.array([int(x) for x in range(800)]).reshape(100, 8)}
        self.label = [0 for _ in range(50)] + [1 for _ in range(50)]
        fea = tf.feature_column.sequence_categorical_column_with_identity(
            key="c1",
            num_buckets=800
        )

        emb = tf.feature_column.embedding_column(
            fea,
            dimension=32)
        feature_columns = [emb]
        self.model = sqlflow_models.StackedBiLSTMClassifier(feature_columns=feature_columns, stack_units=[64, 32])
        self.model_class = sqlflow_models.StackedBiLSTMClassifier


if __name__ == '__main__':
    unittest.main()


