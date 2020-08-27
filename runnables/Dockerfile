FROM sqlflow/sqlflow:step

RUN apt-get clean && apt-get update && \
    apt-get -qq install libmysqld-dev libmysqlclient-dev

ADD ./requirements.txt /
RUN pip3 install --no-cache-dir -r /requirements.txt && rm -rf /requirements.txt

ADD . /opt/sqlflow/run
ENV PYTHONPATH "${PYTHONPATH}:/opt/sqlflow/run"
