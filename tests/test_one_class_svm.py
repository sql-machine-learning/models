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
import shutil
import tempfile
import unittest

import numpy as np
import tensorflow as tf
from sqlflow_models import OneClassSVM


class TestOneClassSVM(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.old_cwd = os.getcwd()
        os.chdir(self.tmp_dir)

    def tearDown(self):
        os.chdir(self.old_cwd)
        shutil.rmtree(self.tmp_dir)

    def create_dataset(self):
        def generator():
            for _ in range(10):
                x1 = np.random.random(size=[1, 1])
                x2 = np.random.random(size=[1, 1])
                yield x1, x2

        def dict_mapper(x1, x2):
            return {"x1": x1, "x2": x2}

        dataset = tf.data.Dataset.from_generator(
            generator, output_types=(tf.dtypes.float32, tf.dtypes.float32))
        return dataset.map(dict_mapper)

    def test_main(self):
        svm = OneClassSVM()
        train_dataset = self.create_dataset()
        svm.sqlflow_train_loop(train_dataset)

        predict_dataset = self.create_dataset()
        for features in predict_dataset:
            pred = svm.sqlflow_predict_one(features)
            pred = np.array(pred)
            self.assertEqual(pred.shape, (1, 1))
            self.assertTrue(pred[0][0] == 1 or pred[0][0] == -1)


if __name__ == '__main__':
    unittest.main()
