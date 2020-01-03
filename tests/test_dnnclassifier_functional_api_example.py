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
        # self.features = {"c1": [float(x) for x in range(100)],
        #                  "c2": [float(x) for x in range(100)],
        #                  "c3": [float(x) for x in range(100)],
        #                  "c4": [float(x) for x in range(100)]}
        # self.label = [0 for _ in range(50)] + [1 for _ in range(50)]
        # feature_columns = [tf.feature_column.numeric_column(key) for key in
        #                    self.features]
        # fieldmetas = {
        #     "c1": {"name": "c1", "shape": [1], "dtype": tf.float32},
        #     "c2": {"name": "c2", "shape": [1], "dtype": tf.float32},
        #     "c3": {"name": "c3", "shape": [1], "dtype": tf.float32},
        #     "c4": {"name": "c4", "shape": [1], "dtype": tf.float32},
        # }
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
