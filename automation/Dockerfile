# https://hub.docker.com/_/ubuntu
FROM ubuntu:22.04

RUN apt-get update && \
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata nano curl python3 python3-pip && \
  apt-get -y upgrade && \
  rm -rf /var/lib/apt/lists/*

# https://github.com/nteract/papermill
# https://docs.celeryq.dev/en/stable/index.html
RUN pip3 install --no-cache papermill celery

# TODO: implement
CMD tail -f /dev/null
