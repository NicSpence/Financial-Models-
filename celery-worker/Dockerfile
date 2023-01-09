# https://hub.docker.com/_/python/
FROM python:3.10

# https://docs.celeryq.dev/en/stable/getting-started/backends-and-brokers/redis.html#broker-redis
RUN pip3 install --no-cache celery[redis]

COPY *.py .

# https://docs.celeryq.dev/en/stable/userguide/periodic-tasks.html#starting-the-scheduler
# if we want to scale this out in the future and have multiple celery workers, we'll
# need to split out the celery beat service
ENTRYPOINT celery -A task worker --autoscale 10 --loglevel=info --beat
