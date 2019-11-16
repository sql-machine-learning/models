FROM sqlflow/modelzoo_base

RUN pip install tensorflow==2.0.0b1 scikit-learn==0.20.0 numpy==1.16.2 pandas==0.25.1
RUN mkdir -p /tmp && cd /tmp && \
git clone https://github.com/sql-machine-learning/elasticdl.git && \
mkdir -p /elasticdl/python/elasticdl/ && \
cp -r elasticdl/elasticdl/python/elasticdl/layers /elasticdl/python/elasticdl/ && \
cd /
ADD *.py /sqlflow_models/
