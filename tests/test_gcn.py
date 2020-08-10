import sqlflow_models
from tests.base import BaseTestCases

import tensorflow as tf
import numpy as np
import unittest
import random


def build_karate_club_graph():
    # All 78 edges are stored in two numpy arrays. One for source endpoints
    # while the other for destination endpoints. 
    # Credit to: https://docs.dgl.ai/tutorials/basics/1_first.html
    src = np.array([1, 2, 2, 3, 3, 3, 4, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 9, 10, 10,
        10, 11, 12, 12, 13, 13, 13, 13, 16, 16, 17, 17, 19, 19, 21, 21,
        25, 25, 27, 27, 27, 28, 29, 29, 30, 30, 31, 31, 31, 31, 32, 32,
        32, 32, 32, 32, 32, 32, 32, 32, 32, 33, 33, 33, 33, 33, 33, 33,
        33, 33, 33, 33, 33, 33, 33, 33, 33, 33])
    dst = np.array([0, 0, 1, 0, 1, 2, 0, 0, 0, 4, 5, 0, 1, 2, 3, 0, 2, 2, 0, 4,
        5, 0, 0, 3, 0, 1, 2, 3, 5, 6, 0, 1, 0, 1, 0, 1, 23, 24, 2, 23,
        24, 2, 23, 26, 1, 8, 0, 24, 25, 28, 2, 8, 14, 15, 18, 20, 22, 23,
        29, 30, 31, 8, 9, 13, 14, 15, 18, 19, 20, 22, 23, 26, 27, 28, 29, 30,
        31, 32])
    u = np.concatenate([src, dst])
    v = np.concatenate([dst, src])
    u = np.expand_dims(u, axis=1)
    v = np.expand_dims(v, axis=1)
    return np.concatenate([u,v], 1)

def acc(y, label):
    '''Function to calculate the accuracy.'''
    ll = tf.equal(tf.argmax(label, -1), tf.argmax(y, -1))
    accuarcy = tf.reduce_mean(tf.cast(ll, dtype=tf.float32))
    return accuarcy

def evaluate(x, y, model):
    '''Function to evaluate the performance of model.'''
    metric = dict()
    y_pred = model.predict(x)
    metric['acc'] = np.round(acc(y, y_pred), 5)
    return metric

class TestGCN(BaseTestCases.BaseTest):
    def setUp(self):
        feature = [[0,1,2]+random.sample(range(3, 20), 8),
                   [0,1,2]+random.sample(range(18, 40),8),
                   [0,1,2]+random.sample(range(38, 60),8),
                   [0,1,2]+random.sample(range(58, 80),8)]
        label = ['Shotokan', 'Gōjū-ryū', 'Wadō-ryū', 'Shitō-ryū']
        nodes = np.array(list(range(34)))
        edges = build_karate_club_graph()
        features, labels = list(), list()
        for i in range(34):
            idx = random.randint(0,3)
            features.append(np.eye(81)[feature[idx]].sum(0))
            labels.append(label[idx])
        self.inputs = [dict() for i in range(len(edges)*2)]
        self.labels = list()
        for i in range(len(edges)):
            self.inputs[i]['id'] = tf.convert_to_tensor(edges[i][0])
            self.inputs[i]['features'] = tf.convert_to_tensor(features[edges[i][0]])
            self.inputs[i]['from_node_id'] = tf.convert_to_tensor(edges[i][0])
            self.inputs[i]['to_node_id'] = tf.convert_to_tensor(edges[i][1])
            self.labels.append(tf.convert_to_tensor([labels[edges[i][0]]]))
        for i in range(len(edges)):
            self.inputs[i+len(edges)]['id'] = tf.convert_to_tensor(edges[i][1])
            self.inputs[i+len(edges)]['features'] = tf.convert_to_tensor(features[edges[i][1]])
            self.inputs[i+len(edges)]['from_node_id'] = tf.convert_to_tensor(edges[i][0])
            self.inputs[i+len(edges)]['to_node_id'] = tf.convert_to_tensor(edges[i][1])
            self.labels.append(tf.convert_to_tensor([labels[edges[i][1]]]))
        self.model = sqlflow_models.GCN(nhid=16, nclass=4, epochs=20, train_ratio=0.2, eval_ratio=0.15)
        self.model_class = sqlflow_models.GCN

    def test_train_and_predict(self):
        self.setUp()
        self.model.compile(optimizer=optimizer(),
                           loss='categorical_crossentropy')
        self.model.sqlflow_train_loop(zip(self.inputs, self.labels))
        metric = evaluate([self.model.features, self.model.adjacency], self.model.labels, self.model)
        assert (metric['acc'] > 0)

def optimizer():
    return tf.keras.optimizers.Adam(lr=0.01)

if __name__ == '__main__':
    unittest.main()


