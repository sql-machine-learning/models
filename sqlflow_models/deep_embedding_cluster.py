#!usr/bin/env python
# -*- coding:utf-8 _*-

"""
__author__ : chenxiang
__email__ : alfredchenxiang@didichuxing.com
__file_name__ : deep_embedding_cluster.py
__create_time__ : 2019/09/03
"""
from datetime import datetime
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.data import make_one_shot_iterator
from tensorflow.python.feature_column.feature_column_v2 import DenseFeatures
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.python.keras.layers import Dense, Layer
from tensorflow.python.keras import backend
import numpy as np
from sklearn.cluster import KMeans
from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD
import pandas as pd


class DeepEmbeddingClusterModel(keras.Model):

    def __init__(self,
                 feature_columns,
                 n_clusters=10,
                 kmeans_init=20,
                 run_pretrain=True,
                 existed_pretrain_model=None,
                 pretrain_dims=None,
                 pretrain_activation_func='relu',
                 pretrain_batch_size=256,
                 train_batch_size=256,
                 pretrain_epochs=1,
                 pretrain_initializer='glorot_uniform',
                 train_max_iters=1000,
                 update_interval=100,
                 tol=0.001,
                 loss=None):
        """
        Implement cluster model mostly based on DEC.
        :param feature_columns:
        :param n_clusters: Number of clusters.
        :param kmeans_init: Number of running K-Means to get best choice of centroids.
        :param run_pretrain: Run pre-train process or not.
        :param existed_pretrain_model: Path of existed pre-train model. Not used now.
        :param pretrain_dims: Dims of layers which is used for build autoencoder.
        :param pretrain_activation_func: Active function of autoencoder layers.
        :param pretrain_batch_size: Size of batch when pre-train.
        :param train_batch_size: Size of batch when run train.
        :param pretrain_epochs: Number of epochs when pre-train.
        :param pretrain_initializer: Initialize function for autoencoder layers.
        :param train_max_iters: Number of iterations when train.
        :param update_interval: Interval between updating target distribution.
        :param tol: tol.
        :param loss: Default 'kld' when init.
        """
        super(DeepEmbeddingClusterModel, self).__init__(name='DECModel')

        # Common
        self._feature_columns = feature_columns
        self._n_clusters = n_clusters
        self._default_loss = loss if loss else 'kld'
        self._train_max_iters = train_max_iters
        self._train_batch_size = train_batch_size
        self._update_interval = update_interval
        self._current_interval = 0
        self._tol = tol

        # Pre-train
        self._run_pretrain = run_pretrain
        self._existed_pretrain_model = existed_pretrain_model
        self._pretrain_activation_func = pretrain_activation_func
        self._pretrain_batch_size = pretrain_batch_size
        self._pretrain_dims = pretrain_dims
        self._pretrain_epochs = pretrain_epochs
        self._pretrain_initializer = pretrain_initializer
        self._pretrain_optimizer = SGD(lr=1, momentum=0.9)

        # K-Means
        self._kmeans_init = kmeans_init

        # Cluster
        self._cluster_optimizer = SGD(lr=0.01, momentum=0.9)

        # Build model
        self._n_stacks = len(self._pretrain_dims)
        self.input_layer = DenseFeatures(feature_columns)

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
        print('{} Start pre_train.'.format(datetime.now()))

        # Concatenate input feature to meet requirement of keras.Model.fit()
        def _concate_generate(dataset_element, label):
            concate_y = tf.stack([dataset_element[feature.key] for feature in self._feature_columns], axis=1)
            return (dataset_element, concate_y)

        y = x.map(map_func=_concate_generate)
        y.prefetch(1)
        print('{} Finished dataset transform.'.format(datetime.now()))

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
        # autoencoder
        self._autoencoder = keras.Sequential(layers=[self.input_layer] + self.encoder_layers + self.decoder_layers,
                                             name='autoencoder')
        self._autoencoder.compile(optimizer=self._pretrain_optimizer, loss='mse')
        # encoder
        self._encoder = keras.Sequential(layers=[self.input_layer] + self.encoder_layers, name='encoder')
        self._encoder.compile(optimizer=self._pretrain_optimizer, loss='mse')

        callbacks = [
            EarlyStopping(monitor='loss', patience=2, min_delta=0.001),
            ReduceLROnPlateau(monitor='loss', factor=0.1, patience=2)
        ]
        print('{} Training auto-encoder.'.format(datetime.now()))
        self._autoencoder.fit_generator(generator=y, epochs=self._pretrain_epochs, callbacks=callbacks)

        # encoded_input
        # type : numpy.ndarray shape : (num_of_all_records,num_of_cluster) (70000,10) if mnist
        print('{} Calculating encoded_input.'.format(datetime.now()))
        self.encoded_input = self._encoder.predict(x)

        del self._autoencoder
        del self._encoder
        del self.decoder_layers
        print('{} Done pre-train.'.format(datetime.now()))

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
        self.y_pred_last = self.kmeans.fit_predict(self.encoded_input)
        print('{} Done init centroids by k-means.'.format(datetime.now()))

    def cluster_train_loop(self, x):
        # Preparation
        ite = make_one_shot_iterator(x)
        features, labels = ite.get_next()
        self.fit(x=features, y=labels)

        # Pre-train autoencoder to prepare weights of encoder layers.
        self.pre_train(x)

        # initialize centroids for clustering.
        self.init_centroids()

        # Setting cluster layer.
        self.get_layer(name='clustering').set_weights([self.kmeans.cluster_centers_])

        # Train
        print('{} Start preparing training dataset.'.format(datetime.now()))
        all_records = {}
        for (feature_dict, label) in x:  # type : dict and EagerTensor
            for feature_name, feature_series in feature_dict.items():  # type : str and EagerTensor
                if feature_name in all_records:
                    all_records[feature_name] = np.concatenate([all_records[feature_name], feature_series])
                else:
                    all_records[feature_name] = feature_series

        all_records_df = pd.DataFrame.from_dict(all_records)
        all_records_ndarray = all_records_df.values
        record_num, feature_num = all_records_df.shape
        print('{} Done preparing training dataset.'.format(datetime.now()))

        index_array = np.arange(record_num)
        index, loss, p = 0, 0., None
        for ite in range(self._train_max_iters):
            if ite % self._update_interval == 0:
                q = self.predict(all_records)  # numpy.ndarray shape(record_num,n_clusters)
                p = self.target_distribution(q)  # update the auxiliary target distribution p
                y_pred = q.argmax(1)
                # delta_percentage means the percentage of changed predictions in this train stage.
                delta_percentage = np.sum(y_pred != self.y_pred_last).astype(np.float32) / y_pred.shape[0]
                print('{} Updating at iter: {} -> delta_percentage: {}.'.format(datetime.now(), ite, delta_percentage))
                self.y_pred_last = np.copy(y_pred)
                if ite > 0 and delta_percentage < self._tol:
                    print('Early stopping since delta_table {} has reached tol {}'.format(delta_percentage, self._tol))
                    break
            idx = index_array[index * self._train_batch_size: min((index + 1) * self._train_batch_size, record_num)]
            loss = self.train_on_batch(x=np.split(all_records_ndarray[idx], feature_num, axis=1), y=p[idx])
            if ite % 100 == 0:
                print('{} Training at iter:{} -> loss:{}.'.format(datetime.now(), ite, loss))
            index = index + 1 if (index + 1) * self._train_batch_size <= record_num else 0  # Update index

    @staticmethod
    def prepare_prediction_column(prediction):
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
        self.kernel = self.add_weight(name='kernel', shape=shape, initializer='glorot_uniform', trainable=True)
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
