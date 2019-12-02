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
            self.model.fit(train_input_fn(self.features, self.label),
                epochs=3,
                steps_per_epoch=200, verbose=0)
            loss, acc = self.model.evaluate(eval_input_fn(self.features, self.label))
            print(loss, acc)
            assert(loss < 10)
            assert(acc > 0.1)
