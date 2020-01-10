import tensorflow as tf
import unittest
import sys

def train_input_fn(features, labels, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    return dataset


def eval_input_fn(features, labels, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    dataset = dataset.batch(batch_size)
    return dataset

class BaseTestCases:
    class BaseTest(unittest.TestCase):
        def setUp(self):
            self.model, self.features, self.label = None, {}, None

        def test_train_and_predict(self):
            self.setUp()
            model_pkg = sys.modules[self.model_class.__module__]
            self.model.compile(optimizer=model_pkg.optimizer(),
                loss=model_pkg.loss,
                metrics=["accuracy"])
            self.history = self.model.fit(train_input_fn(self.features, self.label),
                epochs=10,
                steps_per_epoch=200, 
                verbose=1)
            self.historyloss =  self.history.history['loss']
            loss_decline_rate = (self.historyloss[0] - self.historyloss[-1]) \
                                / self.historyloss[0]
            print('historyloss is {}, and the loss_decline_rate is {}'.\
                format(self.historyloss, loss_decline_rate))
            assert(loss_decline_rate > 0.3)

    class BaseEstimatorTest(BaseTest):
        def test_train_and_predict(self):
            self.setUp()
            input_fn = lambda: train_input_fn(self.features, self.label)
            train_spec = tf.estimator.TrainSpec(input_fn=input_fn, max_steps=1)
            eval_spec = tf.estimator.EvalSpec(input_fn=lambda: eval_input_fn(self.features, self.label))
            baseline = tf.estimator.train_and_evaluate(self.model, train_spec, eval_spec)[0]
            train_spec = tf.estimator.TrainSpec(input_fn=input_fn, max_steps=2000)
            result = tf.estimator.train_and_evaluate(self.model, train_spec, eval_spec)[0]
            loss_decline_rate = 1- result["loss"] / baseline["loss"]
            print('historyloss is {}, and the loss_decline_rate is {}'.\
                format(baseline["loss"], loss_decline_rate))
            assert(loss_decline_rate > 0.3)
