#!usr/bin/env python
# -*- coding:utf-8 _*-

"""
__author__ : chenxiang
__email__ : alfredchenxiang@didichuxing.com
__file_name__ : cluster_model.py
__create_time__ : 2019/09/03
"""
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.layers import Dense, Layer
from tensorflow.python.keras import backend
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
    dims = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((dims, dims), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


class DECModel(keras.Model):
    def __init__(self, n_clusters=10,
                 kmeans_init=20,
                 run_pretrain=False,
                 existed_pretrain_model=None,
                 pretrain_dims=None,
                 pretrain_activation_func='relu',
                 pretrain_batch_size=256,
                 pretrain_epochs=10,
                 pretrain_initializer='glorot_uniform',
                 loss=None):
        """
        Implement cluster model mostly based on DEC.
        :param n_clusters: Number of clusters
        :param kmeans_init: Number of running K-Means to get best choice of centroids.
        :param run_pretrain:
        :param existed_pretrain_model:
        :param pretrain_dims:
        :param pretrain_activation_func:
        :param pretrain_batch_size:
        :param pretrain_epochs:
        """
        super(DECModel, self).__init__(name='DECModel')

        # Common
        self._n_clusters = n_clusters
        self._default_loss = loss if loss else 'kld'

        # Pre-train
        self._run_pretrain = run_pretrain
        self._existed_pretrain_model = existed_pretrain_model
        self._pretrain_activation_func = pretrain_activation_func
        self._pretrain_batch_size = pretrain_batch_size
        self._pretrain_dims = pretrain_dims if pretrain_dims else [784, 500, 500, 2000, 10]
        self._pretrain_epochs = pretrain_epochs
        self._pretrain_initializer = pretrain_initializer
        self._pretrain_optimizer = keras.optimizers.SGD(lr=1, momentum=0.9)

        # K-Means
        self._kmeans_init = kmeans_init

        # Cluster
        self._cluster_optimizer = keras.optimizers.SGD(lr=0.01, momentum=0.9)

        # Build model
        self._n_stacks = len(self._pretrain_dims)
        self.input_layer = keras.layers.InputLayer(input_shape=(self._pretrain_dims[0],), name='model_input')

        # Layers - encoder
        self.encoder_layers = []
        for i in range(self._n_stacks - 1):
            self.encoder_layers.append(Dense(units=self._pretrain_dims[i + 1],
                                             activation=self._pretrain_activation_func,
                                             name='encoder_%d' % i))

        self.encoder_layers.append(Dense(units=self._pretrain_dims[-1],
                                         kernel_initializer=self._pretrain_initializer,
                                         name='encoder_%d' % (self._n_stacks - 1)))

        self.clustering_layer = ClusteringLayer(name='clustering', n_clusters=self._n_clusters)

    def default_optimizer(self):
        return self._cluster_optimizer

    def default_loss(self):
        return self._default_loss

    @staticmethod
    def target_distribution(q):
        """
        Calculate auxiliary softer target distributions by raising q to the second power and
        then normalizing by frequency.
        :param q: Original distributions.
        :return: Auxiliary softer target distributions
        """
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def pre_train(self, x):
        """
        Used for preparing encoder part by loading ready-to-go model or training one.
        :param x:
        :return:
        """
        # Layers - decoder
        self.decoder_layers = []
        for i in range(self._n_stacks - 1, 0, -1):
            self.decoder_layers.append(Dense(units=self._pretrain_dims[i],
                                             activation=self._pretrain_activation_func,
                                             kernel_initializer=self._pretrain_initializer,
                                             name='decoder_%d' % i))

        self.decoder_layers.append(Dense(units=self._pretrain_dims[0],
                                         kernel_initializer=self._pretrain_initializer,
                                         name='decoder_0'))
        # Pretrain - autoencoder, encoder
        self._autoencoder = keras.Sequential(layers=[self.input_layer] + self.encoder_layers + self.decoder_layers,
                                             name='autoencoder')
        self._autoencoder.compile(optimizer=self._pretrain_optimizer, loss='mse')
        self._encoder = keras.Sequential(layers=[self.input_layer] + self.encoder_layers, name='encoder')
        self._encoder.compile(optimizer=self._pretrain_optimizer, loss='mse')

        if self._run_pretrain:
            if self._existed_pretrain_model:
                try:
                    self._autoencoder.load_weights(filepath=self._existed_pretrain_model)
                except FileNotFoundError as fe:
                    print(fe)
                    pass
                except ImportError as ie:
                    print(ie)
                    pass
            else:
                raise ValueError('Model path should be specified when using pre-train model.')

        else:
            self._autoencoder.fit(x, x, batch_size=self._pretrain_batch_size, epochs=self._pretrain_epochs)

        self.encoded_input = self._encoder.predict(x)

        del self._autoencoder
        del self._encoder
        del self.decoder_layers

    def call(self, inputs, training=None, mask=None):
        x = self.input_layer(inputs)
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
        return self.clustering_layer(x)

    def init_centroids(self):
        """
        Training K-means `_kmeans_init` times on the output of encoder to get best initial centroids.
        :return:
        """
        self.kmeans = KMeans(n_clusters=self._n_clusters, n_init=self._kmeans_init)
        self.kmeans.fit_predict(self.encoded_input)

    def evaluate(self, x=None, y=None, batch_size=None, verbose=1, sample_weight=None, steps=None,
                 callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False):
        metric = dict()
        q = self.predict(x)
        y_pred = q.argmax(1)
        metric['acc'] = np.round(acc(y, y_pred), 5)
        metric['nmi'] = np.round(nmi(y, y_pred), 5)
        metric['ari'] = np.round(ari(y, y_pred), 5)
        return q, metric

    def cluster_train_loop(self, x, y, batch_size=256, maxiter=8000, update_interval=150, tol=0.001):
        index_array = np.arange(x.shape[0])
        # initialize centroids for clustering.
        self.init_centroids()
        self.get_layer(name='clustering').set_weights([self.kmeans.cluster_centers_])
        self.display_model_info()
        index, loss, p = 0, 0., None
        y_pred_last = self.kmeans.fit_predict(self.encoded_input)

        for ite in range(int(maxiter)):
            if ite % update_interval == 0:
                q, metric = self.evaluate(x, y)
                p = self.target_distribution(q)  # update the auxiliary target distribution p
                y_pred = q.argmax(1)
                loss = np.round(loss, 5)
                print('Iter %d : acc = %.5f, nmi = %.5f, ari = %.5f, loss = %.5f' % (
                    ite, metric['acc'], metric['nmi'], metric['ari'], loss))

                # delta_label means the percentage of changed predictions in this train stage.
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = np.copy(y_pred)
                if ite > 0 and delta_label < tol:
                    print('Early stopping since delta_table {} has reached tol {}'.format(delta_label, tol))
                    break
            idx = index_array[index * batch_size: min((index + 1) * batch_size, x.shape[0])]
            loss = self.train_on_batch(x=x[idx], y=p[idx])
            index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0

    @staticmethod
    def prepare_prediction_colum(prediction):
        """ Return the cluster label of the highest probability. """
        return prediction.argmax(axis=-1)

    def display_model_info(self, verbose=0):
        if verbose >= 0:
            print('Summary : ')
            print(self.summary())
        if verbose >= 1:
            print('Layer\'s Info : ')
            for layer in self.encoder_layers:
                print(layer.name + ' : ')
                print(layer.get_weights())
            # Cluster
            print(self.clustering_layer.name + ' : ')
            print(self.clustering_layer.get_weights())


class ClusteringLayer(Layer):
    def __init__(self, n_clusters, alpha=1.0, **kwargs):
        """
        Using clustering layer to refine the cluster centroids by learning from current high confidence assignment
        using auxiliary target distribution.

        :param n_clusters: Number of clusters.
        :param weights: Initial cluster centroids.
        :param alpha: Degrees of freedom parameters in Student's t-distribution. Default to 1.0 for all experiments.
        :param kwargs:
        """
        self.n_clusters = n_clusters
        self.alpha = alpha
        super(ClusteringLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[1]
        shape = tf.TensorShape(dims=(self.n_clusters, input_dim))
        self.kernel = self.add_weight(name='kernel',
                                      shape=shape,
                                      initializer='glorot_uniform',
                                      trainable=True)
        super(ClusteringLayer, self).build(shape)

    def call(self, inputs, **kwargs):
        q = 1.0 / (1.0 + (backend.sum(backend.square(backend.expand_dims(inputs, axis=1) - self.kernel),
                                      axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = backend.transpose(backend.transpose(q) / backend.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
