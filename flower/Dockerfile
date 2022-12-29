# https://hub.docker.com/_/python/
FROM python:3.10

RUN apt-get update && \
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends dos2unix && \
  rm -rf /var/lib/apt/lists/*

# https://docs.celeryq.dev/en/stable/getting-started/backends-and-brokers/redis.html#broker-redis
RUN pip3 install --no-cache flower redis

COPY startup.sh /startup.sh
RUN dos2unix /startup.sh
RUN chmod +x /startup.sh

CMD /startup.sh