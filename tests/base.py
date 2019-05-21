import unittest
import tensorflow as tf

def train_input_fn(features, labels, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    return dataset


def prepare_dataset():
    pass


class BaseTestCases:
    class BaseTest(object):
        def setUp(self):
            self.model, self.features, self.label = None, {}, None

        def test_train_and_predict(self):
            self.setUp()
            train_input_fn(self.features, self.label)

            print('Calling BaseTest:testCommon')
            value = 5
            assert(value == 5)
