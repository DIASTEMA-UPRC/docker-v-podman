FROM ubuntu:20.04

EXPOSE 5000

RUN apt update -y
RUN apt upgrade -y
RUN apt install -y python3 python3-pip python-is-python3
RUN pip install -U pip
RUN pip install gunicorn

WORKDIR /app
RUN mkdir -p ./models

ADD requirements.txt .
ADD app.py .
ADD models/model.h5 ./models

RUN pip install -r requirements.txt

ENTRYPOINT gunicorn --bind :5000 app:app
