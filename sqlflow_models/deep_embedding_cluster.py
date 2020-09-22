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
from tensorflow import keras
from tensorflow.python.data import make_one_shot_iterator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Layer, DenseFeatures, InputSpec
from tensorflow.keras import backend
import numpy as np
from sklearn.cluster import KMeans
from tensorflow.keras.losses import kld
from tensorflow.keras.optimizers import SGD
import tensorflow_datasets as tfds
import pandas as pd

_train_lr = 0.01
_default_loss = kld

class DeepEmbeddingClusterModel(keras.Model):

    def __init__(self,
                 feature_columns,
                 n_clusters=10,
                 kmeans_init=20,
                 run_pretrain=True,
                 existed_pretrain_model=None,
                 pretrain_dims=[100, 100, 10],
                 pretrain_activation_func='relu',
                 pretrain_use_callbacks=False,
                 pretrain_cbearlystop_patience=30,
                 pretrain_cbearlystop_mindelta=0.0001,
                 pretrain_cbreduce_patience=10,
                 pretrain_cbreduce_factor=0.1,
                 pretrain_epochs=30,
                 pretrain_initializer='glorot_uniform',
                 pretrain_lr=1,
                 train_lr=0.01,
                 train_max_iters=8000,
                 update_interval=100,
                 train_use_tol=True,
                 tol=0.0001,
                 loss=kld):

        """
        Implement cluster model mostly based on DEC.
        :param feature_columns: a list of tf.feature_column
        :param n_clusters: Number of clusters.
        :param kmeans_init: Number of running K-Means to get best choice of centroids.
        :param run_pretrain: Run pre-train process or not.
        :param existed_pretrain_model: Path of existed pre-train model. Not used now.
        :param pretrain_dims: Dims of layers which is used for build autoencoder.
        :param pretrain_activation_func: Active function of autoencoder layers.
        :param pretrain_use_callbacks: Use callbacks when pre-train or not.
        :param pretrain_cbearlystop_patience: Patience value of EarlyStopping when use callbacks.
        :param pretrain_cbearlystop_mindelta: Min_delta value of EarlyStopping when use callbacks.
        :param pretrain_cbreduce_patience: Patience value of ReduceLROnPlateau when use callbacks.
        :param pretrain_cbreduce_factor: Factor value of ReduceLROnPlateau when use callbacks.
        :param pretrain_epochs: Number of epochs when pre-train.
        :param pretrain_initializer: Initialize function for autoencoder layers.
        :param pretrain_lr: learning rate to train the auto encoder.
        :param train_lr: learning rate to train the cluster network.
        :param train_max_iters: Number of iterations when train.
        :param update_interval: Interval between updating target distribution.
        :param train_use_tol: Use tolerance during clusteringlayer or not.
        :param tol: Tolerance of earlystopping when train during clusteringlayer.
        :param loss: Default 'kld' when init.
        """
        global _train_lr
        global _default_loss
        super(DeepEmbeddingClusterModel, self).__init__(name='DECModel')

        # Common
        self._feature_columns = feature_columns
        self._feature_columns_dims = len(self._feature_columns)
        self._n_clusters = n_clusters
        _default_loss = loss
        self._train_max_iters = train_max_iters
        self._update_interval = update_interval
        self._current_interval = 0
        self._train_use_tol = train_use_tol
        self._tol = tol

        # Pre-train
        self._run_pretrain = run_pretrain
        self._existed_pretrain_model = existed_pretrain_model
        self._pretrain_activation_func = pretrain_activation_func
        self._pretrain_dims = pretrain_dims
        self._pretrain_epochs = pretrain_epochs
        self._pretrain_initializer = pretrain_initializer
        self._pretrain_lr = pretrain_lr
        self._pretrain_optimizer = SGD(lr=self._pretrain_lr, momentum=0.9)

        # Pre-train-callbacks
        self._pretrain_use_callbacks = pretrain_use_callbacks
        self._pretrain_cbearlystop_patience = pretrain_cbearlystop_patience
        self._pretrain_cbearlystop_mindelta = pretrain_cbearlystop_mindelta
        self._pretrain_cbreduce_patience = pretrain_cbreduce_patience
        self._pretrain_cbreduce_factor = pretrain_cbreduce_factor 

        # K-Means
        self._kmeans_init = kmeans_init

        # Cluster
        _train_lr = train_lr
        self._cluster_optimizer = SGD(lr=_train_lr, momentum=0.9)

        # Build model
        self._n_stacks = len(self._pretrain_dims)
        self.input_layer = DenseFeatures(feature_columns)

        # Layers - encoder
        self.encoder_layers = []
        for i in range(self._n_stacks):
            self.encoder_layers.append(Dense(units=self._pretrain_dims[i],
                                             activation=self._pretrain_activation_func,
                                             name='encoder_%d' % i))

        self.clustering_layer = ClusteringLayer(name='clustering', n_clusters=self._n_clusters)

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

        print('{} Start preparing training dataset to save into memory.'.format(datetime.now()))
        # Concatenate input feature to meet requirement of keras.Model.fit()
        def _concate_generate(dataset_element):
            concate_y = tf.stack([dataset_element[feature.key] for feature in self._feature_columns], axis=1)
            return (dataset_element, concate_y)

        y = x.cache().map(map_func=_concate_generate)
        y.prefetch(1)
        
        self.input_x = dict()
        self.input_y = None
        for np_sample in tfds.as_numpy(y):
            sample_dict = np_sample[0]
            label = np_sample[1]
            if self.input_y is None:
                self.input_y = label
            else:
                self.input_y = np.concatenate([self.input_y, label])
            if len(self.input_x) == 0:
                self.input_x = sample_dict
            else:
                for k in self.input_x:
                    self.input_x[k] = np.concatenate([self.input_x[k], sample_dict[k]])
        print('{} Done preparing training dataset.'.format(datetime.now()))

        # Layers - decoder
        self.decoder_layers = []
        for i in range(self._n_stacks - 2, -1, -1):
            self.decoder_layers.append(Dense(units=self._pretrain_dims[i],
                                             activation=self._pretrain_activation_func,
                                             kernel_initializer=self._pretrain_initializer,
                                             name='decoder_%d' % (i + 1)))

        self.decoder_layers.append(Dense(units=self._feature_columns_dims,
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

        # pretrain_callbacks
        print('{} Training auto-encoder.'.format(datetime.now()))
        if self._pretrain_use_callbacks:
            callbacks = [
                EarlyStopping(monitor='loss', 
                    patience=self._pretrain_cbearlystop_patience, min_delta=self._pretrain_cbearlystop_mindelta),
                ReduceLROnPlateau(monitor='loss', 
                    factor=self._pretrain_cbreduce_factor, patience=self._pretrain_cbreduce_patience)
            ]
            self._autoencoder.fit(self.input_x, self.input_y, 
                epochs=self._pretrain_epochs, callbacks=callbacks, verbose=1)
        else:
            self._autoencoder.fit(self.input_x, self.input_y, 
                epochs=self._pretrain_epochs, verbose=1)
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

    def sqlflow_train_loop(self, x, epochs=1, verbose=0):
        """ Parameter `epochs` and `verbose` will not be used in this function. """
        # There is a bug which will cause build failed when using `DenseFeatures` with `keras.Model`
        # https://github.com/tensorflow/tensorflow/issues/28111
        # Using 'predict' to solve this problem here.
        # Preparation
        for features in x.take(1):
            self.predict(x=features)

        # Get train.batch_size from sqlflow
        for feature_name, feature_series in features.items():
            self._train_batch_size = feature_series.shape[0]
            break

        # Pre-train autoencoder to prepare weights of encoder layers.
        self.pre_train(x)

        # Initialize centroids for clustering.
        self.init_centroids()

        # Setting cluster layer.
        self.get_layer(name='clustering').set_weights([self.kmeans.cluster_centers_])

        # Train
        # flatten y to shape (num_samples, flattened_features)
        record_num = self.input_y.shape[0]
        feature_dims = self.input_y.shape[1:]
        feature_dim_total = 1
        for d in feature_dims:
            feature_dim_total = feature_dim_total * d
        y_reshaped = self.input_y.reshape([record_num, feature_dim_total])
        print('{} Done preparing training dataset.'.format(datetime.now()))

        index_array = np.arange(record_num)
        index, loss, p = 0, 0., None
        
        for ite in range(self._train_max_iters):
            if ite % self._update_interval == 0:
                q = self.predict(self.input_x)  # numpy.ndarray shape(record_num,n_clusters)
                p = self.target_distribution(q)  # update the auxiliary target distribution p
                
                if self._train_use_tol:
                    y_pred = q.argmax(1)
                    # delta_percentage means the percentage of changed predictions in this train stage.
                    delta_percentage = np.sum(y_pred != self.y_pred_last).astype(np.float32) / y_pred.shape[0]
                    print('{} Updating at iter: {} -> delta_percentage: {}.'.format(datetime.now(), ite, delta_percentage))
                    self.y_pred_last = np.copy(y_pred)
                    if ite > 0 and delta_percentage < self._tol:
                        print('Early stopping since delta_table {} has reached tol {}'.format(delta_percentage, self._tol))
                        break
            idx = index_array[index * self._train_batch_size: min((index + 1) * self._train_batch_size, record_num)]

            loss = self.train_on_batch(x=list(y_reshaped[idx].T), y=p[idx])
            if ite % 100 == 0:
                print('{} Training at iter:{} -> loss:{}.'.format(datetime.now(), ite, loss))
            index = index + 1 if (index + 1) * self._train_batch_size <= record_num else 0  # Update index

    def display_model_info(self, verbose=0):
        if verbose >= 0:
            print('Summary : ')
            print(self.summary())
        if verbose >= 1:
            print('Layer\'s Shape : ')
            for layer in self.encoder_layers:
                print(layer.name + ' : ')
                for i in layer.get_weights():
                    print(i.shape)
            print(self.clustering_layer.name + ' : ')
            for i in self.clustering_layer.get_weights():
                print(i.shape)
        if verbose >= 2:
            print('Layer\'s Info : ')
            for layer in self.encoder_layers:
                print(layer.name + ' : ')
                print(layer.get_weights())
            # Cluster
            print(self.clustering_layer.name + ' : ')
            print(self.clustering_layer.get_weights())


def optimizer():
    global _train_lr
    return SGD(lr=_train_lr, momentum=0.9)

def loss(labels, output):
    global _default_loss
    return _default_loss(labels, output)

def prepare_prediction_column(prediction):
    """ Return the cluster label of the highest probability. """
    return prediction.argmax(axis=-1)

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
        self.input_spec = InputSpec(ndim=2)
        super(ClusteringLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=backend.floatx(), shape=(None, input_dim))
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
