import tensorflow as tf

_loss = ''

class StackedBiLSTMClassifier(tf.keras.Model):
    def __init__(self, feature_columns, stack_units=[32], hidden_size=64, n_classes=2):
        """StackedBiLSTMClassifier
        :param feature_columns: All columns must be embedding of sequence column with same sequence_length.
        :type feature_columns: list[tf.embedding_column].
        :param stack_units: Units for LSTM layer.
        :type stack_units: vector of ints.
        :param n_classes: Target number of classes.
        :type n_classes: int.
        """
        global _loss
        super(StackedBiLSTMClassifier, self).__init__()

        self.feature_layer = tf.keras.experimental.SequenceFeatures(feature_columns)
        self.stack_bilstm = []
        self.stack_size = len(stack_units)
        self.stack_units = stack_units
        self.n_classes = n_classes
        if self.stack_size > 1:
            for i in range(self.stack_size - 1):
                self.stack_bilstm.append(
                    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.stack_units[i], return_sequences=True))
                )
        self.lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.stack_units[-1]))
        self.hidden = tf.keras.layers.Dense(hidden_size, activation='relu')
        if self.n_classes == 2:
            # special setup for binary classification
            pred_act = 'sigmoid'
            _loss = 'binary_crossentropy'
        else:
            pred_act = 'softmax'
            _loss = 'categorical_crossentropy'
        self.pred = tf.keras.layers.Dense(n_classes, activation=pred_act)

    def call(self, inputs):
        x, seq_len = self.feature_layer(inputs)
        seq_mask = tf.sequence_mask(seq_len)
        if self.stack_size > 1:
            for i in range(self.stack_size - 1):
                x = self.stack_bilstm[i](x, mask=seq_mask)
        x = self.lstm(x, mask=seq_mask)
        x = self.hidden(x)
        return self.pred(x)

def optimizer():
    """Default optimizer name. Used in model.compile."""
    return 'adam'

def loss():
    global _loss
    return _loss

def prepare_prediction_column(prediction):
    """Return the class label of highest probability."""
    return prediction.argmax(axis=-1)
