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


def optimizer():
    return SGD(lr=0.1, momentum=0.9)

def loss():
    return None

class MyScoreCard(keras.Model):

    def __init__(self, feature_columns=None):
        super(MyScoreCard, self).__init__(name='ScoreCard')

        self._factor = 20/np.log(2)
        self._offset = 600 - 20*np.log(20) / np.log(2)

    def call(self):
        pass
        
    def _mono_bin(self, y, x, n=10):
        # population frequency
        r = 0
        bad_num = y.sum()
        good_num = y.count() - y.sum()
        d1 = pd.DataFrame({'x':x,'y':y,'bucket':pd.qcut(x,n, duplicates='drop')})
        d2 = d1.groupby('bucket',as_index=True)
        d3 = pd.DataFrame(d2.x.min(),columns=['min_bin']) 

        d3["min"] = d2.min().x
        d3["max"] = d2.max().x
        d3["badcostum"] = d2.sum().y
        d3["goodcostum"] = d2.count().y - d2.sum().y
        d3["total"] = d2.count().y
        d3["bad_rate"] = d2.sum().y/d2.count().y
        d3["woe"] = np.log(d3["badcostum"]/d3["goodcostum"] * good_num/ bad_num)
        iv = ((d3["badcostum"]/bad_num - d3["goodcostum"]/good_num)*d3["woe"])
        d3["iv"] = iv
        woe = list(d3["woe"].round(6))
        cut = list(d3["max"].round(6))
        cut.insert(0, float("-inf"))
        cut[-1] = float("inf")
        return d3, cut, woe, iv

    def _to_dataframe(self, dataset):
        x_df = pd.DataFrame()
        y_df = pd.DataFrame()

        for features, label in dataset:
            dx = {}
            dy = {}
            for name, value in features.items():
                dx[name] = value.numpy()[0]
            dy['label'] = label.numpy()[0][0]
            x_df = x_df.append(dx, ignore_index=True)
            y_df = y_df.append(dy, ignore_index=True)
        return x_df, y_df

    def _replace_woe(self, x, cut, woe):
        return pd.cut(x, cut, labels=pd.Categorical(woe))

    def _calsumscore(self, woe_list, coe):
        n = coe.shape[1]
        serise = 0
        for i in range(n):
            serise += coe[0][i] * np.array(woe_list.iloc[:, i])
        score = serise * self._factor + self._offset
        return score
    def _get_score(self, coe, woe):
        scores = []
        for w in woe:
            scores.append(round(coe * w * self._factor, 0))
        return scores

    def sqlflow_train_loop(self, x, epochs=1, verbose=0):
        ite = make_one_shot_iterator(x)
        ite.get_next()

        x_df, y_df = self._to_dataframe(x)

        x_train_dict = {}
        woe_dict = {}
        for col in x_df.columns:
            if col in ['id', 'number_of_dependents']:
                continue
            fx1, cut1, x1_woe, iv1 = self._mono_bin(y_df['label'], x_df[col])
            woe_dict[col] = fx1
            x_replaced_woe = self._replace_woe(x_df[col], cut1, x1_woe)
            x_train_dict[col] = x_replaced_woe
        x_train = pd.DataFrame.from_dict(x_train_dict)
        clf = LogisticRegression()
        clf.fit(x_train, y_df['label'])
        clf.predict_proba(x_train)
        coe = clf.coef_
        scores = self._calsumscore(x_train, coe)
        col_i = -1
        for i in range(len(x_df.columns)):
            col_name = x_df.columns[i]
            if col_name in ['id', 'number_of_dependents']:
                continue
            col_i += 1
            col_woe = woe_dict[col_name]
            for j, w in enumerate(col_woe['woe']):
                print(col_name, col_woe['woe'].index.to_list()[j], round(coe[0][col_i] * w * self._factor, 0))
