#!usr/bin/env python  
# -*- coding:utf-8 _*-

""" 
__author__ : chenxiang 
__email__ : alfredchenxiang@didichuxing.com
__file_name__ : dec_demo.py 
__create_time__ : 2019/09/10 
"""
import time

from tensorflow.python import keras
import numpy as np

from sqlflow_models.cluster_keras import DECModel


def data_factory(datasource='mnist'):
    """
    Support three datasources in plan : mnist, reuters and stl-10.
    NOW SUPPORT MNIST ONLY.
    :param datasource:
    :return:
    """
    supported_datasource = ['mnist', 'reuters']
    assert datasource in supported_datasource, 'Sorry, supported datasource : {}'.format(supported_datasource)
    if datasource == 'mnist':
        (train_data, train_labels), (test_data, test_labels) = keras.datasets.mnist.load_data()
        parameters = {'batch_size': 256, 'maxiter': 8000, 'update_interval': 140, 'tol': 0.001}
    elif datasource == 'reuters':
        raise NotImplementedError
    else:
        raise ValueError

    x = np.concatenate((train_data, test_data))
    y = np.concatenate((train_labels, test_labels))
    x = x.reshape((x.shape[0], -1))
    x = np.divide(x, 255.)
    return x, y, parameters


def output_result(datasource, parameters, metric):
    """
    Format output of parameters, metrics.
    :param datasource:
    :param parameters:
    :param metric:
    :return:
    """
    print('-' * 5 + 'Training DEC Model' + '-' * 5)
    print('Using DataSource : {}'.format(datasource))
    print('With Parameters : ')
    for k, v in parameters.items():
        print('\t{} : {}'.format(k, v))
    print('Getting Metrics : ')
    for k, v in metric.items():
        print('\t{} : {}'.format(k, v))


def timelogger(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print('Running Cost {} seconds.'.format(round(time.time() - start, 3)))
        return result
    return wrapper


@timelogger
def train_evaluate(datasource='mnist'):
    # Initial
    x, y, parameters = data_factory(datasource=datasource)
    dec = DECModel()
    dec.compile(loss=dec.default_loss(), optimizer=dec.default_optimizer())
    # Train
    if hasattr(dec, 'pre_train'):
        dec.pre_train(x)

    dec.build(input_shape=x.shape)

    print(dec.display_model_info(verbose=2))

    if hasattr(dec, 'cluster_train_loop'):
        dec.cluster_train_loop(x=x, y=y,
                               batch_size=parameters.get('batch_size', 256),
                               maxiter=parameters.get('maxiter', 8000),
                               update_interval=parameters.get('update_interval', 140),
                               tol=parameters.get('tol', 0.001))

    # Evaluate
    _, metric = dec.evaluate(x=x, y=y)

    # Output
    output_result(datasource, parameters, metric)

    # Save weights
    dec.save_weights('dec',save_format='h5')

    # # Load weights
    #
    # ## Load encoder and train cluster
    # dec2 = DECModel(run_pretrain=True,
    #                 existed_pretrain_model='dec')
    #
    # ##  Predict
    # dec2_result = dec2.predict(x)
    #
    # ## Load whole model
    # dec3 = DECModel(run_pretrain=True)
    # dec3.load_weights(filepath='dec')


if __name__ == '__main__':
    train_evaluate()
