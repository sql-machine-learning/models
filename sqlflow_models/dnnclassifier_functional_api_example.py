import tensorflow as tf

global _loss

def dnnclassifier_functional_model(feature_columns, field_metas, n_classes=2,  learning_rate=0.001):
    feature_layer_inputs = dict()
    for fmkey in field_metas:
        fm = field_metas[fmkey]
        feature_layer_inputs[fm["feature_name"]] = tf.keras.Input(shape=(fm["shape"]), name=fm["feature_name"], dtype=fm["dtype"])
    feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
    feature_layer_outputs = feature_layer(feature_layer_inputs)
    global _loss
    if n_classes == 2:
        # special setup for binary classification
        pred_act = 'sigmoid'
        _loss = 'binary_crossentropy'
    else:
        pred_act = 'softmax'
        _loss = 'categorical_crossentropy'
    x = tf.keras.layers.Dense(128, activation='relu')(feature_layer_outputs)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    pred = tf.keras.layers.Dense(n_classes, activation=pred_act)(x)
    return tf.keras.Model(inputs=[v for v in feature_layer_inputs.values()], outputs=pred)

def loss(labels, output):
    global _loss
    if _loss == "binary_crossentropy":
        return tf.reduce_mean(tf.keras.losses.binary_crossentropy(labels, output))
    elif _loss == "categorical_crossentropy":
        return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, output))

def epochs():
    return 1

def optimizer(lr=0.1):
    return tf.keras.optimizers.Adagrad(lr=lr)

def prepare_prediction_column(self, prediction):
    """Return the class label of highest probability."""
    return prediction.argmax(axis=-1)
