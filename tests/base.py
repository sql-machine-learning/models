import unittest
import tensorflow as tf

def train_input_fn(features, labels, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    return dataset


def prepare_dataset():
    pass


class BaseTestCases:
    class BaseTest(unittest.TestCase):
        def __init__(self):
            super(BaseTestCases.BaseTest, self).__init__()
            self.model, self.features, self.label = None, None, None

        def test_train_and_predict(self):
            train_input_fn(self.features, self.label)

            self
            print('Calling BaseTest:testCommon')
            value = 5
            self.assertEquals(value, 5)

