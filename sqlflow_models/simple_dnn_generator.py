# This file is based on the AdaNet example
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import adanet
import tensorflow as tf

_NUM_LAYERS_KEY = "num_layers"


class _SimpleDNNBuilder(adanet.subnetwork.Builder):
  """Builds a DNN subnetwork for AdaNet."""

  def __init__(self, feature_columns, optimizer, layer_size, num_layers, learn_mixture_weights,
               seed):
    """Initializes a `_DNNBuilder`.

    Args:
      optimizer: An `Optimizer` instance for training both the subnetwork and
        the mixture weights.
      layer_size: The number of nodes to output at each hidden layer.
      num_layers: The number of hidden layers.
      learn_mixture_weights: Whether to solve a learning problem to find the
        best mixture weights, or use their default value according to the
        mixture weight type. When `False`, the subnetworks will return a no_op
        for the mixture weight train op.
      seed: A random seed.

    Returns:
      An instance of `_SimpleDNNBuilder`.
    """

    self._optimizer = optimizer
    self._layer_size = layer_size
    self._num_layers = num_layers
    self._learn_mixture_weights = learn_mixture_weights
    self._feature_columns = feature_columns
    self._seed = seed

  def build_subnetwork(self,
                       features,
                       logits_dimension,
                       training,
                       iteration_step,
                       summary,
                       previous_ensemble=None):
    """See `adanet.subnetwork.Builder`."""

    input_layer = tf.compat.v1.feature_column.input_layer(features, self._feature_columns)
    kernel_initializer = tf.compat.v1.glorot_uniform_initializer(seed=self._seed)
    last_layer = input_layer
    for _ in range(self._num_layers):
      last_layer = tf.compat.v1.layers.dense(
          last_layer,
          units=self._layer_size,
          activation=tf.nn.relu,
          kernel_initializer=kernel_initializer)
    logits = tf.compat.v1.layers.dense(
        last_layer,
        units=logits_dimension,
        kernel_initializer=kernel_initializer)

    persisted_tensors = {_NUM_LAYERS_KEY: tf.constant(self._num_layers)}
    return adanet.Subnetwork(
        last_layer=last_layer,
        logits=logits,
        complexity=self._measure_complexity(),
        persisted_tensors=persisted_tensors)

  def _measure_complexity(self):
    """Approximates Rademacher complexity as the square-root of the depth."""
    return tf.sqrt(tf.cast(self._num_layers, tf.float32))

  def build_subnetwork_train_op(self, subnetwork, loss, var_list, labels,
                                iteration_step, summary, previous_ensemble):
    """See `adanet.subnetwork.Builder`."""
    return self._optimizer.minimize(loss=loss, var_list=var_list)

  def build_mixture_weights_train_op(self, loss, var_list, logits, labels,
                                     iteration_step, summary):
    """See `adanet.subnetwork.Builder`."""

    if not self._learn_mixture_weights:
      return tf.no_op()
    return self._optimizer.minimize(loss=loss, var_list=var_list)

  @property
  def name(self):
    """See `adanet.subnetwork.Builder`."""

    if self._num_layers == 0:
      # A DNN with no hidden layers is a linear model.
      return "linear"
    return "{}_layer_dnn".format(self._num_layers)


class SimpleDNNGenerator(adanet.subnetwork.Generator):
  """Generates a two DNN subnetworks at each iteration.

  The first DNN has an identical shape to the most recently added subnetwork
  in `previous_ensemble`. The second has the same shape plus one more dense
  layer on top. This is similar to the adaptive network presented in Figure 2 of
  [Cortes et al. ICML 2017](https://arxiv.org/abs/1607.01097), without the
  connections to hidden layers of networks from previous iterations.
  """

  def __init__(self, optimizers, feature_columns, layer_size, learn_mixture_weights, seed):
    """Initializes a DNN `Generator`.

    Args:
      optimizers: A defaultdict of string for training both the subnetwork and
        the mixture weights.
      layer_size: Number of nodes in each hidden layer of the subnetwork
        candidates. Note that this parameter is ignored in a DNN with no hidden
        layers.
      learn_mixture_weights: Whether to solve a learning problem to find the
        best mixture weights, or use their default value according to the
        mixture weight type. When `False`, the subnetworks will return a no_op
        for the mixture weight train op.
      seed: A random seed.

    Returns:
      An instance of `Generator`.
    """

    self._seed = seed
    self._optimizers = optimizers
    self._dnn_builder_fn = functools.partial(
        _SimpleDNNBuilder,
        layer_size=layer_size,
        feature_columns=feature_columns,
        learn_mixture_weights=learn_mixture_weights)

  def generate_candidates(self, previous_ensemble, iteration_number,
                          previous_ensemble_reports, all_reports):
    """See `adanet.subnetwork.Generator`."""

    num_layers = 0
    seed = self._seed
    if previous_ensemble:
      num_layers = tf.get_static_value(
          previous_ensemble.weighted_subnetworks[
              -1].subnetwork.persisted_tensors[_NUM_LAYERS_KEY])
    if seed is not None:
      seed += iteration_number
    optimizer = self._optimizers[num_layers + 0]
    return [self._dnn_builder_fn(num_layers=num_layers, optimizer=optimizer, seed=seed),
            self._dnn_builder_fn(num_layers=num_layers + 1, optimizer=optimizer, seed=seed)]
