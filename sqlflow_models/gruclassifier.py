import tensorflow as tf

_loss = ''

class StackedBiGRUClassifier(tf.keras.Model):
    def __init__(self, feature_columns=None, stack_units=[32], hidden_size=64, n_classes=2):
        """StackedBiGRUClassifier
        :param feature_columns: All columns must be embedding of sequence column with same sequence_length.
        :type feature_columns: list[tf.embedding_column].
        :param stack_units: Units for GRU layer.
        :type stack_units: vector of ints.
        :param n_classes: Target number of classes.
        :type n_classes: int.
        """
        global _loss
        super(StackedBiGRUClassifier, self).__init__()

        self.feature_layer = None
        if feature_columns is not None:
            self.feature_layer = tf.keras.experimental.SequenceFeatures(feature_columns)
        self.stack_bigru = []
        self.stack_size = len(stack_units)
        self.stack_units = stack_units
        self.n_classes = n_classes
        if self.stack_size > 1:
            for i in range(self.stack_size - 1):
                self.stack_bigru.append(
                    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(self.stack_units[i], return_sequences=True))
                )
        self.gru = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(self.stack_units[-1]))
        self.hidden = tf.keras.layers.Dense(hidden_size, activation='relu')
        if self.n_classes == 2:
            # special setup for binary classification
            pred_act = 'sigmoid'
            _loss = 'binary_crossentropy'
            n_out = 1
        else:
            pred_act = 'softmax'
            _loss = 'categorical_crossentropy'
            n_out = self.n_classes
        self.pred = tf.keras.layers.Dense(n_out, activation=pred_act)

    def call(self, inputs):
        if self.feature_layer:
            x, seq_len = self.feature_layer(inputs)
        else:
            x, seq_len = inputs
        seq_mask = tf.sequence_mask(seq_len)
        if self.stack_size > 1:
            for i in range(self.stack_size - 1):
                x = self.stack_bigru[i](x, mask=seq_mask)
        x = self.gru(x, mask=seq_mask)
        x = self.hidden(x)
        return self.pred(x)

def optimizer():
    """Default optimizer name. Used in model.compile."""
    return 'adam'

def loss(labels, output):
    global _loss
    if _loss == "binary_crossentropy":
        return tf.reduce_mean(tf.keras.losses.binary_crossentropy(labels, output))
    elif _loss == "categorical_crossentropy":
        return tf.reduce_mean(tf.keras.losses.categorical_crossentropy(labels, output))

def prepare_prediction_column(prediction):
    """Return the class label of highest probability."""
    return prediction.argmax(axis=-1)

def eval_metrics_fn():
    return {
        "accuracy": lambda labels, predictions: tf.equal(
            tf.argmax(predictions, 1, output_type=tf.int32),
            tf.cast(tf.reshape(labels, [-1]), tf.int32),
        )
    }
