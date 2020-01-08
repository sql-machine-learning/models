import tensorflow as tf

class DNNRegressor(tf.keras.Model):
    def __init__(self, feature_columns=None, hidden_units=[100,100]):
        """DNNRegressor
        :param feature_columns: feature columns.
        :type feature_columns: list[tf.feature_column].
        :param hidden_units: number of hidden units.
        :type hidden_units: list[int].
        """
        super(DNNRegressor, self).__init__()
        self.feature_layer = None
        if feature_columns is not None:
            # combines all the data as a dense tensor
            self.feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
        self.hidden_layers = []
        for hidden_unit in hidden_units:
            self.hidden_layers.append(tf.keras.layers.Dense(hidden_unit, activation='relu'))
        self.prediction_layer = tf.keras.layers.Dense(1)

    def call(self, inputs, training=True):
        if self.feature_layer is not None:
            x = self.feature_layer(inputs)
        else:
            x = tf.keras.layers.Flatten()(inputs)
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
        return self.prediction_layer(x)

def optimizer(learning_rate=0.001):
    """Default optimizer name. Used in model.compile."""
    return tf.keras.optimizers.Adagrad(lr=learning_rate)

def loss(labels, output):
    """Default loss function. Used in model.compile."""
    return tf.keras.losses.MSE(labels, output)

def prepare_prediction_column(prediction):
    """Return the prediction directly."""
    return prediction[0]

def eval_metrics_fn():
    return {
        "mse": lambda labels, predictions: tf.reduce_mean(
            tf.pow(
                tf.cast(predictions, tf.float64) - tf.cast(labels, tf.float64), 2)
            )
    }
