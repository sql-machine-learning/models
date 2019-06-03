import tensorflow as tf

class StackedBiLSTMClassifier(tf.keras.Model):
    def __init__(self, feature_columns, units=64, stack_size=1, n_classes=2):
        """StackedBiLSTMClassifier
        :param feature_columns: All columns must be embedding of sequence column with same sequence_length.
        :type feature_columns: list[tf.embedding_column].
        :param units: Units for LSTM layer.
        :type units: int.
        :param stack_size: number of bidirectional LSTM layers in the stack, default 1.
        :type stack_size: int.
        :param n_classes: Target number of classes.
        :type n_classes: int.
        """
        super(StackedBiLSTMClassifier, self).__init__()


        self.feature_layer = tf.keras.experimental.SequenceFeatures(feature_columns)
        self.stack_bilstm = []
        self.stack_size = stack_size
        if stack_size > 1:
            for i in range(stack_size - 1):
                self.stack_bilstm.append(
                    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units, return_sequences=True))
                )
        self.lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units))
        self.pred = tf.keras.layers.Dense(n_classes, activation='softmax')

    def call(self, inputs):
        x, seq_len = self.feature_layer(inputs)
        seq_mask = tf.sequence_mask(seq_len)
        if self.stack_size > 1:
            for i in range(self.stack_size - 1):
                x = self.stack_bilstm[i](x, mask=seq_mask)
        x = self.lstm(x, mask=seq_mask)
        return self.pred(x)

    def default_optimizer(self):
        """Default optimizer name. Used in model.compile."""
        return 'adam'

    def default_loss(self):
        """Default loss function. Used in model.compile."""
        return 'categorical_crossentropy'

    def default_training_epochs(self):
        """Default training epochs. Used in model.fit."""
        return 1

    def prepare_prediction_column(self, prediction):
        """Return the class label of highest probability."""
        return prediction.argmax(axis=-1)

