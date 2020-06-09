import tensorflow as tf

class RNNBasedTimeSeriesModel(tf.keras.Model):

    def __init__(self,
                 feature_columns=None,
                 stack_units=[500, 500],
                 n_in=7,
                 n_out=1,
                 n_features=1, 
                 model_type='rnn'):
        """RNNBasedTimeSeriesModel
        :param feature_columns: All columns must be embedding of sequence column with same sequence_length.
            type feature_columns: list[tf.feature_column.numeric_column].
        :param stack_units: Units for RNN layer.
            type stack_units: vector of ints.
        :param n_in: Size of time window.
            type n_in: int.
        :param n_out: Number of predicted labels.
            type n_out: int.
        :param n_features: number of features in every time window.
            type n_features: int.
        :param model_type: Specific RNN model to be used, which can be chose from: ('rnn', 'lstm' and 'gru').
            type model_type: string.
        """
        super(RNNBasedTimeSeriesModel, self).__init__(name='RNN_TS_Model')
        # Common
        self.feature_columns = feature_columns
        self.loss = loss
        self.n_out = n_out
        self.n_in = n_in
        self.n_features = n_features
        self.stack_units = stack_units
        self.models = {'rnn':tf.keras.layers.SimpleRNN, 'lstm':tf.keras.layers.LSTM, 'gru':tf.keras.layers.GRU}
        # combines all the data as a dense tensor
        self.feature_layer = None
        if feature_columns is not None:
            self.feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
        self.stack_layers = []
        for unit in self.stack_units[:-1]:
            self.stack_layers.append(self.models[model_type.lower()](unit, input_shape=(self.n_in, self.n_features), return_sequences=True))
        self.rnn = self.models[model_type.lower()](self.stack_units[-1], input_shape=(self.n_in, self.n_features))
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.prediction_layer = tf.keras.layers.Dense(self.n_out)

    def call(self, inputs):
        if self.feature_layer:
            x = self.feature_layer(inputs)
        else:
            x = inputs
        x = tf.reshape(x, (-1, self.n_in, self.n_features))
        for i in range(len(self.stack_units) - 1):
            x = self.stack_layers[i](x)
        x = self.rnn(x)
        x = self.dropout(x)
        return self.prediction_layer(x)

def optimizer(learning_rate=0.001):
    """Default optimizer name. Used in model.compile."""
    return tf.keras.optimizers.Adam(lr=learning_rate)

def prepare_prediction_column(prediction):
    """Return the prediction directly."""
    return prediction

def loss(labels, output):
    return tf.reduce_mean(tf.keras.losses.MSE(labels, output))

