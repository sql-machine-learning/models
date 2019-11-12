import tensorflow as tf

class DNNClassifier(tf.keras.Model):
    def __init__(self, feature_columns, hidden_units=[10,10], n_classes=2):
        """DNNClassifier
        :param feature_columns: feature columns.
        :type feature_columns: list[tf.feature_column].
        :param hidden_units: number of hidden units.
        :type hidden_units: list[int].
        :param n_classes: List of hidden units per layer.
        :type n_classes: int.
        """
        super(DNNClassifier, self).__init__()

        # combines all the data as a dense tensor
        self.feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
        self.hidden_layers = []
        for hidden_unit in hidden_units:
            self.hidden_layers.append(tf.keras.layers.Dense(hidden_unit))
        self.prediction_layer = tf.keras.layers.Dense(n_classes, activation='softmax')

    def call(self, inputs):
        x = self.feature_layer(inputs)
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
        return self.prediction_layer(x)

def optimizer(learning_rate=0.1):
    """Default optimizer name. Used in model.compile."""
    return tf.keras.optimizers.Adagrad(lr=learning_rate)

def loss():
    """Default loss function. Used in model.compile."""
    return 'sparse_categorical_crossentropy'

def prepare_prediction_column(self, prediction):
    """Return the class label of highest probability."""
    return prediction.argmax(axis=-1)