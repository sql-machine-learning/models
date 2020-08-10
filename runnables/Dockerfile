FROM sqlflow/sqlflow:step

RUN apt-get clean && apt-get update && \
    apt-get -qq install python3-pip libmysqld-dev libmysqlclient-dev && \
    ln -s /usr/bin/pip3 /usr/bin/pip

ADD ./requirements.txt /
RUN pip3 install --no-cache-dir -r /requirements.txt && rm -rf /requirements.txt

ADD ./ts_feature_extract /opt/sqlflow/run
ENV PYTHONPATH "${PYTHONPATH}:/opt/sqlflow/run"
