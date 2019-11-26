#!/bin/bash

elasticdl train --image_base=sqlflow/sqlflow_models \
--model_def=dnnclassifier.DNNClassifier \
--training_data=sqlflow_test_iris_train \
--data_reader_params='columns=["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]' \
--envs="ODPS_PROJECT_NAME=gomaxcompute_driver_w7u,ODPS_ACCESS_ID=$ODPS_ACCESS_ID,ODPS_ACCESS_KEY=$ODPS_ACCESS_KEY" \
--minibatch_size=32 \
--num_epochs=2 \
--model_zoo=/sqlflow_models \
--job_name=test-odps \
--num_minibatches_per_task=2 \
--image_pull_policy=Never \
--num_workers=2 \
--master_resource_request="cpu=400m,memory=256Mi" \
--master_resource_limit="cpu=1,memory=2048Mi" \
--worker_resource_request="cpu=400m,memory=256Mi" \
--worker_resource_limit="cpu=1,memory=3072Mi" \
--grads_to_wait=2 \
--output=model_output
