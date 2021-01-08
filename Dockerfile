FROM python:3.6.12-slim-buster

RUN apt-get update && apt-get upgrade -y 

ADD . /app/

WORKDIR /app

RUN pip install -r requirements_docker.txt

EXPOSE 8080

CMD python3 app/app.py
