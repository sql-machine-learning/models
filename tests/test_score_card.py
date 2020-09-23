from sqlflow_models import ScoreCard
import unittest
import tensorflow as tf
from datetime import datetime, timedelta
import numpy as np


class TestScoreCard(unittest.TestCase):
    def create_dataset(self):
        samples = 20
        f = [np.random.randint(20, size=1) for i in range(samples)]
        label = [np.random.randint(2, size=1) for i in range(samples)]

        def generator():
            for i, item in enumerate(f):
                yield [f[i]], label[i]

        def dict_mapper(feature, label):
            return {'f1': feature}, label

        dataset = tf.data.Dataset.from_generator(
            generator, output_types=(tf.dtypes.float32, tf.dtypes.float32)
        )
        dataset = dataset.map(dict_mapper)
        return dataset

    def test_train(self):
        dataset = self.create_dataset()
        m = ScoreCard(pf_bin_size=2)
        m.sqlflow_train_loop(dataset)


if __name__ == '__main__':
    unittest.main()
