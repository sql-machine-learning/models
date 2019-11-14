FROM sqlflow/modelzoo_base

RUN pip install tensorflow==2.0.0b1 scikit-learn==0.20.0 numpy==1.16.2 pandas==0.25.1
ADD *.py /sqlflow_models/
RUN mkdir -p /tmp && cd /tmp && \
git clone https://github.com/sql-machine-learning/elasticdl.git && \
cp -r elasticdl/elasticdl/python/elasticdl/layers /usr/local/lib/python3.7/site-packages/elasticdl-develop-py3.7.egg/elasticdl/python/elasticdl/ && \
cd /
