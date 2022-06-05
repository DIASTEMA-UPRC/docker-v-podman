FROM python:3.8

EXPOSE 5000

RUN apt update -y
RUN apt upgrade -y
RUN pip install -U pip
RUN pip install gunicorn

WORKDIR /app
RUN mkdir -p ./models

ADD requirements.txt .
ADD app.py .
ADD models/model.h5 ./models

RUN pip install -r requirements.txt

ENTRYPOINT gunicorn --bind :5000 app:app
