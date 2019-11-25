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
git checkout 1e5aa1e7bdbbc198c7ecff31dc84884f67235c43 && \
pip install -r elasticdl/requirements.txt && \
python setup.py install && \
cd .. && rm -rf elasticdl
