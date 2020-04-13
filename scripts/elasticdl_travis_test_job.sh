if [ "$SQLFLOW_TEST_DB_MAXCOMPUTE_AK" = "" ] || [ "$SQLFLOW_TEST_DB_MAXCOMPUTE_SK" == "" ]; then
  echo "skip maxcompute test because the env SQLFLOW_TEST_DB_MAXCOMPUTE_AK or SQLFLOW_TEST_DB_MAXCOMPUTE_SK is empty"
  exit 0
fi

curl -s https://raw.githubusercontent.com/sql-machine-learning/elasticdl/4a995fe7eaf91bc5a9d50181e9aaaa14d15c8a09/scripts/setup_k8s_env.sh | bash
kubectl apply -f https://raw.githubusercontent.com/sql-machine-learning/elasticdl/develop/elasticdl/manifests/examples/elasticdl-rbac.yaml

docker run --rm -it --net=host \
      -v $HOME/.kube:/root/.kube \
      -v /home/$USER/.minikube/:/home/$USER/.minikube/ \
      -v /var/run/docker.sock:/var/run/docker.sock \
      -v $PWD:/workspace \
      -e ODPS_ACCESS_ID=$MAXCOMPUTE_AK \
      -e ODPS_ACCESS_KEY=$MAXCOMPUTE_SK \
      sqlflow/sqlflow_models bash /workspace/scripts/test_elasticdl_submit.sh

docker run --rm -it --net=host \
      -v $HOME/.kube:/root/.kube \
      -v /home/$USER/.minikube/:/home/$USER/.minikube/ \
      sqlflow/sqlflow_models \
      bash -c "curl -s https://raw.githubusercontent.com/sql-machine-learning/elasticdl/62b255a918df5b6594c888b19aebbcc74bbce6e4/scripts/validate_job_status.py | python - odps 1 2"
