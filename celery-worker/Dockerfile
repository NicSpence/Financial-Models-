# https://hub.docker.com/_/python/
FROM python:3.10

# https://docs.celeryq.dev/en/stable/getting-started/backends-and-brokers/redis.html#broker-redis
RUN pip3 install --no-cache celery[redis]

COPY *.py .

ENTRYPOINT celery -A task worker --autoscale 10 --loglevel=info
