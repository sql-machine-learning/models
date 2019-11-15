FROM python:3.7

# install PAI python support
RUN pip install pypai

# install go needed by installing ElasticDL
ENV GOPATH /root/go
ENV PATH /usr/local/go/bin:$GOPATH/bin:$PATH
RUN curl --silent https://dl.google.com/go/go1.13.4.linux-amd64.tar.gz | tar -C /usr/local -xzf -

# install ElasticDL to manage ElasticDL jobs
RUN git clone https://github.com/sql-machine-learning/elasticdl.git && \
cd elasticdl && \
git checkout 8676ff3b0a4160df3861d680803bc02412f175ec && \
pip install -r elasticdl/requirements.txt && \
python setup.py install && \
cd .. && rm -rf elasticdl
