import tensorflow as tf

class RawDNNClassifier(tf.keras.Model):
    def __init__(self, hidden_units=[100,100], n_classes=3):
        super(RawDNNClassifier, self).__init__()
        self.feature_layer = None
        self.n_classes = n_classes
        self.hidden_layers = []
        for hidden_unit in hidden_units:
            self.hidden_layers.append(tf.keras.layers.Dense(hidden_unit, activation='relu'))
        if self.n_classes == 2:
            pred_act = 'sigmoid'
            n_out = 1
        else:
            pred_act = 'softmax'
            n_out = self.n_classes
        self.prediction_layer = tf.keras.layers.Dense(n_out, activation=pred_act)

    def call(self, inputs, training=True):
        if self.feature_layer is not None:
            x = self.feature_layer(inputs)
        else:
            x = tf.keras.layers.Flatten()(inputs)
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
        return self.prediction_layer(x)
