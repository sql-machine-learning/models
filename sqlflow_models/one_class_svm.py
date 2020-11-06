# Copyright 2020 The SQLFlow Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pickle

import numpy as np
import tensorflow as tf
from sklearn.svm import OneClassSVM as SklearnOneClassSVM

MODEL_DIR = "model_save"
MODEL_PATH = MODEL_DIR + "/one_class_svm_model"

ENABLE_EAGER_EXECUTION = False

try:
    tf.enable_eager_execution()
    ENABLE_EAGER_EXECUTION = True
except Exception:
    try:
        tf.compat.v1.enable_eager_execution()
        ENABLE_EAGER_EXECUTION = True
    except Exception:
        ENABLE_EAGER_EXECUTION = False


def dataset_reader(dataset):
    if ENABLE_EAGER_EXECUTION:
        for features in dataset:
            yield features
    else:
        iter = dataset.make_one_shot_iterator()
        one_element = iter.get_next()
        with tf.Session() as sess:
            try:
                while True:
                    yield sess.run(one_element)
            except tf.errors.OutOfRangeError:
                pass


class OneClassSVM(tf.keras.Model):
    def __init__(self,
                 feature_columns=None,
                 kernel='rbf',
                 degree=3,
                 gamma='scale',
                 coef0=0.0,
                 tol=0.001,
                 nu=0.5,
                 shrinking=True,
                 cache_size=200,
                 verbose=False,
                 max_iter=-1):
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, "rb") as f:
                self.svm = pickle.load(f)
        else:
            self.svm = SklearnOneClassSVM(kernel=kernel,
                                          degree=degree,
                                          gamma=gamma,
                                          coef0=coef0,
                                          tol=tol,
                                          nu=nu,
                                          shrinking=shrinking,
                                          cache_size=cache_size,
                                          verbose=verbose,
                                          max_iter=max_iter)

    def concat_features(self, features):
        assert isinstance(features, dict)
        each_feature = []
        for k, v in features.items():
            if ENABLE_EAGER_EXECUTION:
                v = v.numpy()
            each_feature.append(v)
        return np.concatenate(each_feature, axis=1)

    def sqlflow_train_loop(self, dataset):
        X = []
        for features in dataset_reader(dataset):
            X.append(self.concat_features(features))
        X = np.concatenate(X)

        self.svm.fit(X)

        if not os.path.exists(MODEL_DIR):
            os.mkdir(MODEL_DIR)

        with open(MODEL_PATH, "wb") as f:
            pickle.dump(self.svm, f, protocol=2)

    def sqlflow_predict_one(self, features):
        features = self.concat_features(features)
        pred = self.svm.predict(features)
        score = self.svm.decision_function(features)
        return pred, score
