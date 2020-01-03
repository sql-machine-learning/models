import sqlflow_models
from tests.base import BaseTestCases

import tensorflow as tf
import unittest

from sklearn.datasets import load_iris


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
        x, y = load_iris(return_X_y=True)
        feature_column_names = ['col_{}'.format(d) for d in range(x.shape[1])]
        self.features = {}
        for feature_name, feature_values in zip(feature_column_names, list(x.T)):
            self.features[feature_name] = feature_values
        self.label = y
        feature_columns = [tf.feature_column.numeric_column(key) for key in self.features]
        fieldmetas = {
            "col_0": {"name": "col_0", "shape": [1], "dtype": tf.float32},
            "col_1": {"name": "col_1", "shape": [1], "dtype": tf.float32},
            "col_2": {"name": "col_2", "shape": [1], "dtype": tf.float32},
            "col_3": {"name": "col_3", "shape": [1], "dtype": tf.float32},
        }
        print(fieldmetas,'********', feature_columns)

        self.model = sqlflow_models.dnnclassifier_functional_model(feature_columns=feature_columns, field_metas=fieldmetas, n_classes=3)
        self.model_class = sqlflow_models.dnnclassifier_functional_model


if __name__ == '__main__':
    unittest.main()
