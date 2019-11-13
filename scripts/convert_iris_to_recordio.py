import os
import shutil
import urllib.request
import tempfile

import tensorflow as tf
import recordio

label_dict = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}

def download_iris(output_dir):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    urllib.request.urlretrieve(url, os.path.join(output_dir, "iris.data"))

def convert():
    tmp = tempfile.mkdtemp()
    download_iris(tmp)
    fn = open(os.path.join(tmp, "iris.data"))
    w = recordio.Writer("iris.recordio")
    for line in fn.readlines():
        striped_line = line.strip("\n")
        if striped_line == "":
            continue
        fields = striped_line.split(",")
        print(fields)
        
        w.write(
            tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            # sepal_length, sepal_width, petal_length, petal_width
                            "sepal_length": tf.train.Feature(
                                float_list=tf.train.FloatList(
                                    value=[float(fields[0])]
                                )
                            ),
                            "sepal_width": tf.train.Feature(
                                float_list=tf.train.FloatList(
                                    value=[float(fields[1])]
                                )
                            ),
                            "petal_length": tf.train.Feature(
                                float_list=tf.train.FloatList(
                                    value=[float(fields[2])]
                                )
                            ),
                            "petal_width": tf.train.Feature(
                                float_list=tf.train.FloatList(
                                    value=[float(fields[3])]
                                )
                            ),
                            "label": tf.train.Feature(
                                int64_list=tf.train.Int64List(
                                    value=[label_dict[fields[4]]]
                                )
                            ),
                        }
                    )
                ).SerializeToString()
        )

    w.close()
    fn.close()
    shutil.rmtree(tmp)

if __name__ == "__main__":
    convert()