#!usr/bin/env python
# -*- coding:utf-8 _*-

"""
__author__ : chenxiang
__email__ : alfredchenxiang@didichuxing.com
__file_name__ : cluster_model.py
__create_time__ : 2019/09/03
"""
import time

import keras.backend as K
import keras
from keras.datasets import mnist
from keras.engine import Layer, InputSpec
from keras.layers import Dense
from keras.optimizers import SGD
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.utils.linear_assignment_ import linear_assignment

ari = adjusted_rand_score
nmi = normalized_mutual_info_score


def acc(y_true, y_pred):
    """
    Calculate clustering accuracy.
    Using the Hungarian algorithm to solve linear assignment problem.
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


class DECModel(object):
    def __init__(self, n_clusters=10, kmeans_init=20, use_pretrain=False, encoder_model_path=None,
                 encoder_dims=None, activation_func='relu', batch_size=256, pretrain_epochs=10):
        super(DECModel, self).__init__()
        self._use_pretrain = use_pretrain
        self._encoder_model_path = encoder_model_path
        self._encoder_dims = encoder_dims if encoder_dims else [784, 500, 500, 2000, 10]
        self._n_clusters = n_clusters
        self._activation_func = activation_func
        self._batch_size = batch_size
        self._pretrain_epochs = pretrain_epochs
        self._kmeans_init = kmeans_init
        self._pretrain_optimizer = SGD(lr=1, momentum=0.9)
        self._optimizer = SGD(lr=0.01, momentum=0.9)
        self._autoencoder, self._encoder = build_autoencoder(dims=self._encoder_dims,
                                                             act=self._activation_func,
                                                             init='glorot_uniform')

    def build_model(self, x):
        self.prepare_encoder(x)
        self._clustering_layer = ClusteringLayer(n_clusters=self._n_clusters, name='clustering')(self._encoder.output)
        self.model = keras.Model(inputs=self._encoder.input, outputs=self._clustering_layer)
        self.model.compile(optimizer=self._optimizer, loss='kld')

    def prepare_encoder(self, x):
        if self._use_pretrain:
            if self._encoder_model_path:
                try:
                    self._encoder.load_weights(filepath=self._encoder_model_path)
                except FileNotFoundError as fe:
                    pass
                except ImportError as ie:
                    pass
            else:
                raise ValueError('Model path should be specified when using pre-train model.')

        else:
            self._autoencoder.compile(optimizer=self._pretrain_optimizer, loss='mse')
            self._autoencoder.fit(x, x, batch_size=self._batch_size, epochs=self._pretrain_epochs)

    def init_centroids(self, x):
        self.kmeans = KMeans(n_clusters=self._n_clusters, n_init=self._kmeans_init)
        self._y_pred = self.kmeans.fit_predict(self._encoder.predict(x))

    def evaluate(self, x, y):
        metric = dict()
        q = self.model.predict(x)
        y_pred = q.argmax(1)
        metric['acc'] = np.round(acc(y, y_pred), 5)
        metric['nmi'] = np.round(nmi(y, y_pred), 5)
        metric['ari'] = np.round(ari(y, y_pred), 5)
        return metric

    def train_on_loop(self, x, y, batch_size=256, maxiter=8000, update_interval=150, tol=0.001):
        self.build_model(x)
        index_array = np.arange(x.shape[0])
        self.init_centroids(x)
        self.model.get_layer(name='clustering').set_weights([self.kmeans.cluster_centers_])
        index, loss = 0, 0.
        y_pred_last = self.kmeans.cluster_centers_

        for ite in range(int(maxiter)):
            if ite % update_interval == 0:
                q = self.model.predict(x)
                p = target_distribution(q)  # update the auxiliary target distribution p

                # evaluate the clustering performance
                y_pred = q.argmax(1)
                if y is not None:
                    metric_acc = np.round(acc(y, y_pred), 5)
                    metric_nmi = np.round(nmi(y, y_pred), 5)
                    metric_ari = np.round(ari(y, y_pred), 5)
                    loss = np.round(loss, 5)
                    print('Iter %d : acc = %.5f, nmi = %.5f, ari = %.5f, loss = %.5f' % (
                        ite, metric_acc, metric_nmi, metric_ari, loss))

                # check stop criterion - model convergence
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = np.copy(y_pred)
                if ite > 0 and delta_label < tol:
                    print('Early stopping since delta_table {} has reached tol {}'.format(delta_label, tol))
                    break
            idx = index_array[index * batch_size: min((index + 1) * batch_size, x.shape[0])]
            loss = self.model.train_on_batch(x=x[idx], y=p[idx])
            index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0


def build_autoencoder(dims, act='relu', init='glorot_uniform'):
    """
    Build autoencoder and encoder simultaneously.
    :param dims: Dimensions of input.
    :param act: Activation method.
    :param init: Initialize method of weights in layers.
    :return: AutoEncoder model and encode part as another model.
    """
    n_stacks = len(dims)
    model_input = keras.Input(shape=(dims[0],), name='model_input')

    # Encode
    x = model_input
    for i in range(n_stacks - 1):
        x = Dense(units=dims[i + 1], activation=act, name='encoder_%d' % i)(x)
    encoded = Dense(units=dims[-1], kernel_initializer=init, name='encoder_%d' % (n_stacks - 1))(x)

    # Decode
    x = encoded
    for i in range(n_stacks - 1, 0, -1):
        x = Dense(units=dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(x)

    x = Dense(units=dims[0], kernel_initializer=init, name='decoder_0')(x)
    decoded = x

    autoencoder = keras.Model(inputs=model_input, outputs=decoded, name='autoencoder')
    encoder = keras.Model(inputs=model_input, outputs=encoded, name='encoder')
    return autoencoder, encoder


class ClusteringLayer(Layer):
    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        """
        Using clustering layer to refine the cluster centroids by learning from current high confidence assignment using auxiliary target distribution.

        :param n_clusters: Number of clusters.
        :param weights: Initial cluster centroids.
        :param alpha: Degrees of freedom parameters in Student's t-distribution. Default to 1.0 for all experiments.
        :param kwargs:
        """
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight((self.n_clusters, input_dim),
                                        initializer='glorot_uniform',
                                        name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def target_distribution(q):
    """
    Calculate auxiliary softer target distributions by raising q to the second power and then normalizing by frequency.
    :param q: Original distributions.
    :return: Auxiliary softer target distributions
    """
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T


def data_factory(datasource='mnist'):
    """
    Support three datasources in plan : mnist, reuters and stl-10.
    NOW SUPPORT MNIST ONLY.
    :param datasource:
    :return:
    """
    supported_datasource = ['mnist', 'reuters']
    assert datasource in supported_datasource, 'Sorry, supported datasource : {}'.format(supported_datasource)
    if datasource == 'mnist':
        (train_data, train_labels), (test_data, test_labels) = mnist.load_data()
        parameters = {'batch_size': 256, 'maxiter': 8000, 'update_interval': 140, 'tol': 0.001}
    elif datasource == 'reuters':
        raise NotImplementedError
    else:
        raise ValueError

    x = np.concatenate((train_data, test_data))
    y = np.concatenate((train_labels, test_labels))
    x = x.reshape((x.shape[0], -1))
    x = np.divide(x, 255.)
    return x, y, parameters


def output_result(datasource, parameters, metric):
    """
    Format output of parameters, metrics.
    :param datasource:
    :param parameters:
    :param metric:
    :return:
    """
    print('-' * 5 + 'Training DEC Model' + '-' * 5)
    print('Using DataSource : {}'.format(datasource))
    print('With Parameters : ')
    for k, v in parameters.items():
        print('\t{} : {}'.format(k, v))
    print('Getting Metrics : ')
    for k, v in metric.items():
        print('\t{} : {}'.format(k, v))


def timelogger(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print('Running Cost {} seconds.'.format(round(time.time() - start, 3)))
        return result

    return wrapper


@timelogger
def train_evaluate(datasource='mnist'):
    # Initial
    x, y, parameters = data_factory(datasource=datasource)
    dec = DECModel()

    # Train
    dec.train_on_loop(x=x, y=y,
                      batch_size=parameters.get('batch_size', 256),
                      maxiter=parameters.get('maxiter', 8000),
                      update_interval=parameters.get('update_interval', 140),
                      tol=parameters.get('tol', 0.001))

    # Evaluate
    metric = dec.evaluate(x=x, y=y)

    # Output
    output_result(datasource, parameters, metric)


if __name__ == '__main__':
    train_evaluate()
    # -----Training DEC Model-----
    # Using DataSource : mnist
    # With Parameters :
    # 	batch_size : 256
    # 	maxiter : 8000
    # 	update_interval : 140
    # 	tol : 0.001
    # Getting Metrics :
    # 	acc : 0.73057
    # 	nmi : 0.69247
    # 	ari : 0.59528
    # Running Cost 1662.565 seconds.
