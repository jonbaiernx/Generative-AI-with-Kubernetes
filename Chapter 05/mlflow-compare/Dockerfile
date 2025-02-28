FROM python:3.10-slim
WORKDIR /mlflow-compare
COPY ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt
COPY . .
CMD gpt2-runs.py