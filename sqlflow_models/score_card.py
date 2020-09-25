#!/bin/env python

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.data import make_one_shot_iterator
from tensorflow.keras.losses import kld
from tensorflow.keras.optimizers import SGD
import numpy as np
import pandas as pd
import scipy.stats.stats as stats
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, auc
import pickle


def optimizer():
    return None


def loss():
    return None


class ScoreCard(keras.Model):

    def __init__(self, feature_columns=None, pf_bin_size=5):
        super(ScoreCard, self).__init__(name='ScoreCard')

        self._target_score = 600
        self._factor = 20/np.log(2)
        self._offset = 600 - 20*np.log(20) / np.log(2)
        self._bins = dict()
        self._pf_bin_size = pf_bin_size

    def _pf_bin(self, y, x):
        # population frequency bucket
        bad_num = y.sum()
        good_num = y.count() - y.sum()
        d1 = pd.DataFrame({'x': x,'y': y,'bucket': pd.qcut(x, self._pf_bin_size, duplicates='drop')})
        d2 = d1.groupby('bucket',as_index=True)
        d3 = pd.DataFrame(d2.x.min(),columns=['min_bin']) 

        d3["min"] = d2.min().x
        d3["max"] = d2.max().x
        d3["badcostum"] = d2.sum().y
        d3["goodcostum"] = d2.count().y - d2.sum().y
        d3["total"] = d2.count().y
        d3["bad_rate"] = d2.sum().y/d2.count().y
        d3["woe"] = np.log(d3["badcostum"]/d3["goodcostum"]*good_num/bad_num)
        iv = ((d3["badcostum"]/bad_num-d3["goodcostum"]/good_num)*d3["woe"])
        d3["iv"] = iv
        woe = list(d3["woe"].round(6))
        cut = list(d3["max"].round(6))
        cut.insert(0, float("-inf"))
        cut[-1] = float("inf")
        return d3, cut, woe, iv

    def _to_dataframe(self, dataset):
        x_df = pd.DataFrame()
        y_df = pd.DataFrame()
        for _, minibatch in enumerate(dataset):
            data, label = minibatch
            dx = {}
            dy = {}
            for name, value in data.items():
                dx[name] = value.numpy()[0][0]
            x_df = x_df.append(dx, ignore_index=True)
            dy['label'] = label.numpy()[0]
            y_df = y_df.append(dy, ignore_index=True)
        return x_df, y_df

    def _replace_woe(self, x, cut, woe):
        return pd.cut(x, cut, labels=pd.Categorical(woe))

    def _woe_encoder(self, x, y):
        x_train_dict = {}
        for col in x.columns:
            dfx, cut, woe, iv = self._pf_bin(y, x[col])
            self._bins[col] = (dfx, cut, woe, iv)
            # replacing by the WOE encode
            x_train_dict[col] = self._replace_woe(x[col], cut, woe)
        return pd.DataFrame.from_dict(x_train_dict)

    def sqlflow_train_loop(self, dataset, epochs=1, verbose=0):
        x_df, y_df = self._to_dataframe(dataset)
        x = self._woe_encoder(x_df, y_df['label'])
        x.to_csv("/tmp/train_woe.csv")
        lr = LogisticRegression()

        x_train, x_test, y_train, y_test = train_test_split(x, y_df['label'])
        lr.fit(x_train, y_train)
        prob = lr.predict_proba(x_test)[:, 1]
        auc_score = roc_auc_score(y_test, prob)
        print("AUC: {}\n".format(auc_score))

        # print the score card
        print("TARGET SCORE: %d" % self._target_score)
        coe = lr.coef_
        for i, col_name in enumerate(x_df.columns):
            bin_cols = self._bins[col_name][0].index.to_list()
            for j, w in enumerate(self._bins[col_name][2]):
                print(col_name, bin_cols[j],
                      round(coe[0][i] * w * self._factor +
                            self._offset/self._pf_bin_size, 0))
