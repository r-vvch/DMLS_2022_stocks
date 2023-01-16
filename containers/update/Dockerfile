FROM ubuntu:20.04 

RUN apt-get update && apt-get install cron -y && apt-get install -y software-properties-common

RUN apt-get install -y iputils-ping

RUN ln -s /usr/bin/python3 /usr/bin/python

ADD update.py /app/update.py

ADD crontab /etc/cron.d/cron_file

RUN chmod 0644 /etc/cron.d/cron_file

RUN crontab /etc/cron.d/cron_file

ENTRYPOINT ["cron", "-f"]