from sqlflow_models import ARIMAWithSTLDecomposition
import unittest
import tensorflow as tf
from datetime import datetime, timedelta
import numpy as np

class TestARIMAWithSTLDecompose(unittest.TestCase):
    def setUp(self):
        self.order = [7, 0, 2]
        self.period = [7, 30]
        self.date_format = '%Y-%m-%d'
        self.train_start = '2014-04-01'
        self.train_end = '2014-08-31'
        self.forecast_start = '2014-09-01'
        self.forecast_end = '2014-09-30'

    def str2datetime(self, date_str):
        if isinstance(date_str, bytes):
            date_str = date_str.decode('utf-8')
        return datetime.strptime(str(date_str), self.date_format)

    def datetime2str(self, date):
        return datetime.strftime(date, self.date_format)

    def create_dataset(self):
        def generator():
            start_date = self.str2datetime(self.train_start)
            end_date = self.str2datetime(self.train_end)
            delta = timedelta(days=1)
            while start_date <= end_date:
                date_str = np.array(self.datetime2str(start_date))
                label = np.random.random(size=[1]) * 1e8
                yield date_str, label
                start_date += delta

        def dict_mapper(date_str, label):
            return {'time': date_str}, label

        dataset = tf.data.Dataset.from_generator(
            generator, output_types=(tf.dtypes.string, tf.dtypes.float32)
        )
        dataset = dataset.map(dict_mapper)
        return dataset

    def prediction_days(self):
        pred_start = self.str2datetime(self.forecast_start)
        pred_end = self.str2datetime(self.forecast_end)
        return (pred_end - pred_start).days + 1

    def test_main(self):
        model = ARIMAWithSTLDecomposition(order=[7, 0, 2],
                                      period=[7, 30],
                                      date_format=self.date_format,
                                      forecast_start=self.forecast_start,
                                      forecast_end=self.forecast_end)
        prediction = model.sqlflow_train_loop(self.create_dataset())
        self.assertEqual(len(prediction), self.prediction_days())


if __name__ == '__main__':
    unittest.main()
