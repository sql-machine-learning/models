from __future__ import absolute_import, division, print_function, unicode_literals
from collections import defaultdict

import absl
import logging
import tensorflow as tf
import warnings

absl.logging.set_verbosity(absl.logging.ERROR)
tf.get_logger().setLevel(logging.ERROR)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
warnings.warn = lambda *args, **kargs:None
import adanet

from tensorflow import keras
from tensorflow_estimator.python.estimator.canned import optimizers
from .simple_dnn_generator import SimpleDNNGenerator


LEARN_MIXTURE_WEIGHTS=True
RANDOM_SEED = 42

class AutoClassifier(adanet.Estimator):
    def __init__(self, feature_columns, layer_size=50, optimizer='Adagrad', linear_optimizer='Ftrl',
                 model_dir=None, n_classes=2, activation_fn=tf.nn.relu, complexity_penalty=0.01,
                 search_every_n_steps=1000, max_iterations=10, config=None):
        """AutoClassifier
        :param feature_columns: Feature columns.
        :type feature_columns: list[tf.feature_column].
        :param layer_size: Number of hidden_units in each layers.
        :type layer_size: int.
        :param n_classes: Number of label classes. Defaults to 2, namely binary classification.
        :type n_classes: int.
        :param optimizer: Optimizer for the the neural multi-layer parts of the generated network.
        :type optimizer: str.
        :param linear_optimizer: Optimizer for the linear part of the generated network.
        :type linear_optimizer: str.
        :param model_dir: Directory to save or restore model checkpoints. 
        :type model_dir: str.
        :param activation_fn: Activation function. 
        :type activation_fn: function.
        :param complexity_penalty: Regularization of the complexity of the network.
        :type complexity_penalty: float.
        :param search_every_n_steps: Search new architecture every n steps.
        :type search_every_n_steps: int.
        :param max_iterations: Max times of architecture searching.
        :type max_iterations: int.
        :param config: Estimator configuration.
        :type config: dict.
        """
        if n_classes == 2:
            head = tf.estimator.BinaryClassHead()
        else:
            head = tf.estimator.MultiClassHead(n_classes=n_classes)

        opts= defaultdict(lambda: optimizers.get_optimizer_instance(optimizer, 0.001))
        opts[0] = optimizers.get_optimizer_instance(linear_optimizer, 0.1)
        # Define the generator, which defines the search space of subnetworks
        # to train as candidates to add to the final AdaNet model.
        subnetwork_generator = SimpleDNNGenerator(
            feature_columns=feature_columns,
            layer_size=layer_size,
            optimizers=opts,
            learn_mixture_weights=LEARN_MIXTURE_WEIGHTS,
            seed=RANDOM_SEED)
        super(AutoClassifier, self).__init__(head=head,
                                             model_dir=model_dir,
                                             adanet_lambda=complexity_penalty,
                                             subnetwork_generator=subnetwork_generator,
                                             max_iteration_steps=search_every_n_steps,
                                             max_iterations=max_iterations)

class AutoRegressor(adanet.Estimator):
    def __init__(self, feature_columns, layer_size=50, optimizer='Adagrad', linear_optimizer='Ftrl',
                 model_dir=None, activation_fn=tf.nn.relu, complexity_penalty=0.01,
                 search_every_n_steps=1000, max_iterations=10, config=None):
        """AutoRegressor
        :param feature_columns: Feature columns.
        :type feature_columns: list[tf.feature_column].
        :param layer_size: Number of hidden_units in each layers.
        :type layer_size: int.
        :param optimizer: Optimizer for the the neural multi-layer parts of the generated network.
        :type optimizer: str.
        :param linear_optimizer: Optimizer for the linear part of the generated network.
        :type linear_optimizer: str.
        :param model_dir: Directory to save or restore model checkpoints. 
        :type model_dir: str.
        :param activation_fn: Activation function. 
        :type activation_fn: function.
        :param complexity_penalty: Regularization of the complexity of the network.
        :type complexity_penalty: float.
        :param search_every_n_steps: Search new architecture every n steps.
        :type search_every_n_steps: int.
        :param max_iterations: Max times of architecture searching.
        :type max_iterations: int.
        :param config: Estimator configuration.
        :type config: dict.
        """
        head = tf.estimator.RegressionHead()

        opts= defaultdict(lambda: optimizers.get_optimizer_instance(optimizer, 0.001))
        opts[0] = optimizers.get_optimizer_instance(linear_optimizer, 0.1)
        # Define the generator, which defines the search space of subnetworks
        # to train as candidates to add to the final AdaNet model.
        subnetwork_generator = SimpleDNNGenerator(
            feature_columns=feature_columns,
            layer_size=layer_size,
            optimizers=opts,
            learn_mixture_weights=LEARN_MIXTURE_WEIGHTS,
            seed=RANDOM_SEED)
        super(AutoRegressor, self).__init__(head=head,
                                            model_dir=model_dir,
                                            adanet_lambda=complexity_penalty,
                                            subnetwork_generator=subnetwork_generator,
                                            max_iteration_steps=search_every_n_steps,
                                            max_iterations=max_iterations)
