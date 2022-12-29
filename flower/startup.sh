#!/bin/bash

set -o errexit
set -o nounset

worker_ready() {
    #celery -A task inspect ping
    celery inspect ping
}

until worker_ready; do
  >&2 echo 'Celery workers not available'
  sleep 1
done
>&2 echo 'Celery worker(s) are available'

# celery -A task  \
#     --broker="${CELERY_BROKER_URL}" \
#     flower
celery \
    --broker="${CELERY_BROKER_URL}" \
    flower