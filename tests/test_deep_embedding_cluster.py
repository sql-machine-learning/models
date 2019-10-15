from tensorflow.python.keras.losses import kld

import sqlflow_models
from tests.base import BaseTestCases, eval_input_fn

import tensorflow as tf
import unittest
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.utils.linear_assignment_ import linear_assignment
import numpy as np
from tensorflow.python import keras


def train_input_fn(features, labels, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    dataset = dataset.shuffle(1000).repeat(1).batch(batch_size)
    return dataset


ari = adjusted_rand_score
nmi = normalized_mutual_info_score


def acc(y_true, y_pred):
    """
    Calculate clustering accuracy.
    Using the Hungarian algorithm to solve linear assignment problem.
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    dims = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((dims, dims), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def evaluate(x, y, model):
    metric = dict()
    q = model.predict(x)
    y_pred = q.argmax(1)
    metric['acc'] = np.round(acc(y, y_pred), 5)
    metric['nmi'] = np.round(nmi(y, y_pred), 5)
    metric['ari'] = np.round(ari(y, y_pred), 5)
    return metric


class TestDeepEmbeddingCluster(BaseTestCases.BaseTest):
    def setUp(self):
        (train_data, train_labels), (test_data, test_labels) = keras.datasets.mnist.load_data()
        x = np.concatenate((train_data, test_data))
        y = np.concatenate((train_labels, test_labels))
        x = x.reshape((x.shape[0], -1))
        x = np.divide(x, 255.)
        # Sample
        x = x[:100]
        y = y[:100]
        # Generate Data
        feature_num = x.shape[1]
        feature_column_names = ['col_{}'.format(d) for d in range(feature_num)]

        self.features = {}
        for feature_name, feature_values in zip(feature_column_names, list(x.T)):
            self.features[feature_name] = feature_values

        # print(self.features)

        self.label = y
        feature_columns = [tf.feature_column.numeric_column(key) for key in self.features]
        pretrain_dims = [500, 500, 2000, 10]
        # Init model
        self.model = sqlflow_models.DeepEmbeddingClusterModel(feature_columns=feature_columns,
                                                              n_clusters=10,
                                                              kmeans_init=20,
                                                              run_pretrain=True,
                                                              existed_pretrain_model=None,
                                                              pretrain_dims=pretrain_dims,
                                                              pretrain_activation_func='relu',
                                                              pretrain_batch_size=256,
                                                              train_batch_size=256,
                                                              pretrain_epochs=10,
                                                              pretrain_initializer='glorot_uniform',
                                                              train_max_iters=100,
                                                              update_interval=20,
                                                              tol=0.001,
                                                              loss=kld)

    def test_train_and_predict(self):
        self.setUp()

        self.model.compile(optimizer=self.model.default_optimizer(),
                           loss=self.model.default_loss())
        self.model.sqlflow_train_loop(train_input_fn(self.features, self.label))
        metric = evaluate(x=eval_input_fn(self.features, self.label), y=self.label, model=self.model)
        print(metric)
        assert (metric['acc'] > 0)


if __name__ == '__main__':
    unittest.main()
