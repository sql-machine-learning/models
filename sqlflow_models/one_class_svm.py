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

MODEL_PATH = "one_class_svm_model"


class OneClassSVM(tf.keras.Model):
    def __init__(self, feature_columns=None, **kwargs):
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, "rb") as f:
                self.svm = pickle.load(f)
        else:
            self.svm = SklearnOneClassSVM(**kwargs)

    def concat_features(self, features):
        assert isinstance(features, dict)
        each_feature = []
        for _, v in features.items():
            each_feature.append(v.numpy())
        return np.concatenate(each_feature, axis=1)

    def sqlflow_train_loop(self, dataset):
        X = []
        for features in dataset:
            X.append(self.concat_features(features))
        X = np.concatenate(X)

        self.svm.fit(X)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(self.svm, f, protocol=2)

    def sqlflow_predict_one(self, features):
        features = self.concat_features(features)
        pred = self.svm.predict(features)
        return [pred]
