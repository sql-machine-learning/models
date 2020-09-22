import sqlflow_models
from tests.base import BaseTestCases

import tensorflow as tf
import numpy as np
import unittest


class TestStackedRNNClassifier(BaseTestCases.BaseTest):
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
        self.model = sqlflow_models.StackedRNNClassifier(feature_columns=feature_columns, stack_units=[64, 32], model_type='rnn')
        self.model_class = sqlflow_models.StackedRNNClassifier

class TestStackedBiRNNClassifier(BaseTestCases.BaseTest):
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
        self.model = sqlflow_models.StackedRNNClassifier(feature_columns=feature_columns, stack_units=[64, 32], model_type='rnn', bidirectional=True)
        self.model_class = sqlflow_models.StackedRNNClassifier

class TestStackedLSTMClassifier(BaseTestCases.BaseTest):
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
        self.model = sqlflow_models.StackedRNNClassifier(feature_columns=feature_columns, stack_units=[64, 32], model_type='lstm')
        self.model_class = sqlflow_models.StackedRNNClassifier

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
        self.model = sqlflow_models.StackedRNNClassifier(feature_columns=feature_columns, stack_units=[64, 32], model_type='lstm', bidirectional=True)
        self.model_class = sqlflow_models.StackedRNNClassifier

class TestStackedGRUClassifier(BaseTestCases.BaseTest):
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
        self.model = sqlflow_models.StackedRNNClassifier(feature_columns=feature_columns, stack_units=[64, 32], model_type='gru')
        self.model_class = sqlflow_models.StackedRNNClassifier

class TestStackedBiGRUClassifier(BaseTestCases.BaseTest):
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
        self.model = sqlflow_models.StackedRNNClassifier(feature_columns=feature_columns, stack_units=[64, 32], model_type='gru', bidirectional=True)
        self.model_class = sqlflow_models.StackedRNNClassifier

if __name__ == '__main__':
    unittest.main()


