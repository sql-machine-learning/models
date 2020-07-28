import tensorflow as tf
import random
import numpy as np

class CustomClassifier(tf.keras.Model):
    def __init__(self, feature_columns=None):
        """The model init function. You can define any model parameter in the function's argument list.
           You can also add custom training routines together with a Keras
           model (see deep_embedding_cluster.py), or define a model with out Keras layers
           (e.g. use sklearn or numpy only).
        """
        pass

    def sqlflow_train_loop(self, x):
        """The custom model traininig loop, input x is a tf.dataset object that generates training data.
        """
        pass
    
    def sqlflow_predict_one(self, sample):
        """Run prediction with one sample and return the prediction result. The result must be a
           list of numpy array. SQLFlow determine the output type by:
           - if the array have only one element, the model must be regression model.
           - if the array have multiple elements:
             - if the sum of all the elements are close to 1, it is likely to be a classification model.
             - else the model is a regression model with multiple outputs.
        """
        pos = random.random()
        neg = 1 - pos
        array = np.array([pos, neg])
        return [array]

    def sqlflow_evaluate_loop(self, x, metric_names):
        """Run evaluation on the validation dataset and return a list of metrics.
           NOTE: the first result metric is always loss. If no loss is defined, add 0.
        """
        metric_len = len(metric_names)
        result = []
        for i in range(metric_len+1):
            result.append(random.random())
        return result
