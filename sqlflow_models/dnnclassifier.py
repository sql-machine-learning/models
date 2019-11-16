import tensorflow as tf

class DNNClassifier(tf.keras.Model):
    def __init__(self, feature_columns=None, hidden_units=[10,10], n_classes=2):
        """DNNClassifier
        :param feature_columns: feature columns.
        :type feature_columns: list[tf.feature_column].
        :param hidden_units: number of hidden units.
        :type hidden_units: list[int].
        :param n_classes: List of hidden units per layer.
        :type n_classes: int.
        """
        super(DNNClassifier, self).__init__()
        self.feature_layer = None
        if feature_columns is not None:
            # combines all the data as a dense tensor
            self.feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
        self.hidden_layers = []
        for hidden_unit in hidden_units:
            self.hidden_layers.append(tf.keras.layers.Dense(hidden_unit))
        self.prediction_layer = tf.keras.layers.Dense(n_classes, activation='softmax')

    def call(self, inputs):
        if self.feature_layer is not None:
            x = self.feature_layer(inputs)
        else:
            x = inputs
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
        return self.prediction_layer(x)

def optimizer(learning_rate=0.1):
    """Default optimizer name. Used in model.compile."""
    return tf.keras.optimizers.Adagrad(lr=learning_rate)

def loss():
    """Default loss function. Used in model.compile."""
    return 'sparse_categorical_crossentropy'

def prepare_prediction_column(prediction):
    """Return the class label of highest probability."""
    return prediction.argmax(axis=-1)

# iris_dataset_fn is only used to test using this model in ElasticDL.
def iris_dataset_fn(dataset, mode, metadata):
    def _parse_data(record):
        label_col_name = "class"
        record = tf.strings.to_number(record, tf.float32)

        def _get_features_without_labels(
            record, label_col_ind, features_shape
        ):
            features = [
                record[:label_col_ind],
                record[label_col_ind + 1 :],  # noqa: E203
            ]
            features = tf.concat(features, -1)
            return tf.reshape(features, features_shape)

        features_shape = (4, 1)
        labels_shape = (1,)
        if mode != Mode.PREDICTION:
            if label_col_name not in metadata.column_names:
                raise ValueError(
                    "Missing the label column '%s' in the retrieved "
                    "ODPS table." % label_col_name
                )
            label_col_ind = metadata.column_names.index(label_col_name)
            labels = tf.reshape(record[label_col_ind], labels_shape)
            return (
                _get_features_without_labels(
                    record, label_col_ind, features_shape
                ),
                labels,
            )
        else:
            if label_col_name in metadata.column_names:
                label_col_ind = metadata.column_names.index(label_col_name)
                return _get_features_without_labels(
                    record, label_col_ind, features_shape
                )
            else:
                return tf.reshape(record, features_shape)

    dataset = dataset.map(_parse_data)

    if mode == Mode.TRAINING:
        dataset = dataset.shuffle(buffer_size=200)
    return dataset
