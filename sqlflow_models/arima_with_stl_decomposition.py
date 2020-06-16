import numpy as np
import six
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import STL
from datetime import datetime
import tensorflow as tf
import pandas as pd

class ARIMAWithSTLDecomposition(tf.keras.Model):
    def __init__(self,
                 order,
                 period,
                 date_format,
                 forecast_start,
                 forecast_end,
                 **kwargs):
        super(ARIMAWithSTLDecomposition, self).__init__()

        self.order = order
        if not isinstance(period, (list, tuple)):
            period = period
        self.period = period
        self.date_format = date_format
        self.forecast_start = self._str2date(forecast_start)
        self.forecast_end = self._str2date(forecast_end)
        self.seasonal = []
        self.kwargs = kwargs

    def _str2date(self, date_str):
        if isinstance(date_str, bytes):
            date_str = date_str.decode('utf-8')
        return datetime.strptime(str(date_str), self.date_format)

    def _read_all_data(self, dataset):
        data = None
        for batch_idx, items in enumerate(dataset):
            if data is None:
                data = [[] for _ in six.moves.range(len(items))]

            for i, item in enumerate(items):
                if isinstance(item, dict):
                    assert len(item) == 1
                    dict_values = list(item.values())
                    item = dict_values[0]

                if isinstance(item, tf.Tensor):
                    item = item.numpy()

                item = np.reshape(item, [-1]).tolist()
                data[i].extend(item)

        dates, values = data
        sorted_dates_index = sorted(range(len(dates)), key=lambda k: dates[k])
        dates = np.array([self._str2date(dates[i]) for i in sorted_dates_index])
        values = np.array([values[i] for i in sorted_dates_index]).astype('float32')

        return dates, values

    def _stl_decompose(self, values):
        left_values = values
        self.seasonal = []
        for p in self.period:
            stl_model = STL(left_values, period=p).fit()
            seasonal = np.array(stl_model.seasonal)
            self.seasonal.append(seasonal)
            left_values -= seasonal

        return left_values

    def _addup_seasonal(self, dates, values):
        time_interval = dates[1] - dates[0]
        start_interval = self.forecast_start - dates[0]
        start_index = int(start_interval.total_seconds() / time_interval.total_seconds())

        length = len(values)

        for p, seasonal in six.moves.zip(self.period, self.seasonal):
            if length % p == 0:
                offset = length
            else:
                offset = (int(length / p) + 1) * p

            idx = start_index - offset
            values += seasonal[idx:idx+length]

        return values

    def _normalize(self, values):
        min_value = np.min(values)
        max_value = np.max(values)
        values = (values - min_value) / (max_value - min_value)
        return values, min_value, max_value
  
    def print_prediction_result(self, prediction, interval):
        t_strs = []
        for i, p in enumerate(prediction):
            t = self.forecast_start + i * interval 
            t_str = datetime.strftime(t, self.date_format)
            t_strs.append(t_str)

        df = pd.DataFrame(data={'time': t_strs, 'prediction': prediction})
        with pd.option_context('display.max_columns', None):
            print(df)

    def sqlflow_train_loop(self, dataset):
        dates, values = self._read_all_data(dataset)

        left_values = self._stl_decompose(values)
        left_values, min_value, max_value = self._normalize(left_values)

        model = ARIMA(left_values, order=self.order, dates=dates).fit(disp=-1)

        prediction = model.predict(start=self.forecast_start, end=self.forecast_end, typ='levels')

        prediction = prediction * (max_value - min_value) + min_value
        prediction = self._addup_seasonal(dates, prediction)
        self.print_prediction_result(prediction, interval=dates[1] - dates[0])
        return prediction

def loss(*args, **kwargs):
    return None

def optimizer(*args, **kwargs):
    return None
