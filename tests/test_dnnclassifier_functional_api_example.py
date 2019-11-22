import sqlflow_models
from tests.base import BaseTestCases

import tensorflow as tf
import unittest
import sys


def train_input_fn(features, labels, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    return dataset


def eval_input_fn(features, labels, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    dataset = dataset.batch(batch_size)
    return dataset
    
class TestDNNClassifier(BaseTestCases.BaseTest):
    def setUp(self):
        self.features = {"c1": [float(x) for x in range(100)],
                         "c2": [float(x) for x in range(100)],
                         "c3": [float(x) for x in range(100)],
                         "c4": [float(x) for x in range(100)]}
        self.label = [0 for _ in range(50)] + [1 for _ in range(50)]
        feature_columns = [tf.feature_column.numeric_column(key) for key in
                           self.features]
        fieldmetas = {
            "c1": {"name": "c1", "shape": [1], "dtype": tf.float32},
            "c2": {"name": "c2", "shape": [1], "dtype": tf.float32},
            "c3": {"name": "c3", "shape": [1], "dtype": tf.float32},
            "c4": {"name": "c4", "shape": [1], "dtype": tf.float32},
        }
        self.model = sqlflow_models.dnnclassifier_functional_model(feature_columns=feature_columns, field_metas=fieldmetas)
        self.model_class = sqlflow_models.dnnclassifier_functional_model


if __name__ == '__main__':
    unittest.main()
