# How to Contribute SQLFLow Models

This guide will introduce how to contribute to SQLFlow models. You can find design doc: [Define SQLFLow Models](/doc/customized+model.md), and feel free to check it out.

## Develop an SQLFlow Model

1. Open the [SQLFlow models repo](https://github.com/sql-machine-learning/models) on your web browser, and fork the official repo to your account.

1. Clone the forked repo on your hosts:

    ``` bash
    > git clone https://github.com/<Your Github ID>/models.git
    ```

1. Set up your local python environment by `make setup && source venv/bin/activate`. If you are using [PyCharm](https://www.jetbrains.com/pycharm/), you can simply `make setup` and then import the `models` folder as a new project.

1. You can add a new mode definition Python script under the folder [sqlflow_models](/sqlflow_models). For example, adding a new Python script `mydnnclassfier.py`:

    ``` text
    `-sqlflow_models
        |- dnnclassifier.py
        `- mydnnclassifier.py
    ```

1. You can choose whatever name you like for your model. Your model definition should be a [keras subclass model](https://keras.io/models/about-keras-models/#model-subclassing)

    ``` python
    import tensorflow as tf

    class MyDNNClassifier(tf.keras.Model):
        def __init__(self, feature_columns, hidden_units=[10,10], n_classes=2):
            ...
            ...
    ```

1. Import `MyDNNClassfier` in [sqlflow_models/\_\_init__.py](/sqlflow_models/__init__.py):

    ``` python
    ...
    from .mydnnclassfier import MyDNNClassifier
    ```

1. You can test your `MyDNNClassifier` by adding a new Python unit test script `tests/test_mydnnclassifier.py` and run the test as: `python tests/test_mydnnclassifier.py`:

    ``` python
    from  sqlflow_models import MyDNNClassifier
    from tests.base import BaseTestCases

    import tensorflow as tf
    import unittest


    class TestMyDNNClassifier(BaseTestCases.BaseTest):
        def setUp(self):
            self.features = {...}
            self.label = [...]
            feature_columns = [...]
            self.model = MyDNNClassifier(feature_columns=feature_columns)

    if __name__ == '__main__':
        unittest.main()
    ```

## Test Your SQLFlow Model

If you have developed a new model, please perform the integration test with the SQLFlow gRPC server to make sure it works well with SQLFlow.

1. Launch an SQLFlow all-in-one Docker container

    ``` bash
    cd ./models
    > docker run --rm -it -v $PWD:/models -p 8888:8888 sqlflow/sqlflow
    ```

1. Update `sqlflow_models` in the SQLFlow all-in-one Docker container:

    ``` bash
    > docker exec -it <container-id> pip install -U /models
    ```

1. Open a web browser and go to `localhost:8888` to access the Jupyter Notebook. Using your custom model by modifying the `TRAIN` parameter of the SQLFlow extend SQL: `TRAIN sqlflow_models.MyDNNClassifier`:

``` sql
SELECT * from iris.train
TRAIN sqlflow_models.MyDNNClassifier
WITH n_classes = 3, hidden_units = [10, 20]
COLUMN sepal_length, sepal_width, petal_length, petal_width
LABEL class
INTO sqlflow_models.my_dnn_model;
```

## Publish your model in the SQLFlow all-in-one Docker image

If you have already tested your code, please create a pull request and invite other develops to review it. If one of the develops **approve** your pull request, then you can merge it to the develop branch.
The travis-ci would build the SQLFlow all-in-one Docker image with the latest models code every night and push it to the Docker hub with tag: `sqlflow/sqlflow:nightly`, you can find the latest models in it the second day.
