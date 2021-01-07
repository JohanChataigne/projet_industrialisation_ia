FROM python:3.6.12-slim-buster

RUN apt-get update && apt-get upgrade -y 

ADD . /app/

WORKDIR /app

RUN pip install -r requirements.txt

CMD python3 app.py
