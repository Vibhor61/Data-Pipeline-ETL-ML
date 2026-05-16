FROM python:3.10-slim

WORKDIR /mlflow

COPY requirements/mlflow.txt /requirements.txt

RUN pip install --default-timeout=100 --no-cache-dir -r /requirements.txt

EXPOSE 5000