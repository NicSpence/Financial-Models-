from celery import Celery
import os
from celery.schedules import crontab

app = Celery(
    'postman',
    broker=os.environ.get('CELERY_BROKER_URL'),
    backend=os.environ.get('CELERY_RESULT_BACKEND')
)

@app.on_after_configure.connect
def setup_periodic_tasks(sender, **kwargs):
    # Executes every hour
    # https://docs.celeryq.dev/en/stable/userguide/periodic-tasks.html#crontab-schedules
    sender.add_periodic_task(
        crontab(minute=0, hour='*/1'),
        helloworld.s('Hello World!'),
        name="hello world"
    )

@app.task
def helloworld(arg):
    print(arg)