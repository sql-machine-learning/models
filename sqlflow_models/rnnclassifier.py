import tensorflow as tf

_loss = ''

class StackedRNNClassifier(tf.keras.Model):
    def __init__(self, feature_columns=None, stack_units=[32], hidden_size=64, n_classes=2, model_type='rnn', bidirectional=False):
        """StackedRNNClassifier
        :param feature_columns: All columns must be embedding of sequence column with same sequence_length.
        :type feature_columns: list[tf.embedding_column].
        :param stack_units: Units for RNN layer.
        :type stack_units: vector of ints.
        :param n_classes: Target number of classes.
        :type n_classes: int.
        :param model_type: Specific RNN model to be used, which can be chose from: ('rnn', 'lstm' and 'gru').
        :type model_type: string.
        :param bidirectional: Whether to use bidirectional or not.
        :type bidirectional: bool.
        """
        global _loss
        super(StackedRNNClassifier, self).__init__()

        self.models = {'rnn':tf.keras.layers.SimpleRNN, 'lstm':tf.keras.layers.LSTM, 'gru':tf.keras.layers.GRU}
        self.bidirectionals = {True: tf.keras.layers.Bidirectional, False: lambda x: x}
        self.feature_layer = None
        if feature_columns is not None:
            self.feature_layer = tf.keras.experimental.SequenceFeatures(feature_columns)
        self.stack_rnn = []
        self.stack_size = len(stack_units)
        self.stack_units = stack_units
        self.n_classes = n_classes
        if self.stack_size > 1:
            for i in range(self.stack_size - 1):
                self.stack_rnn.append(
                    self.bidirectionals[bidirectional](self.models[model_type.lower()](self.stack_units[i], return_sequences=True))
                )
        self.rnn = self.bidirectionals[bidirectional](self.models[model_type.lower()](self.stack_units[-1]))
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
                x = self.stack_rnn[i](x, mask=seq_mask)
        x = self.rnn(x, mask=seq_mask)
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
