FROM python:3.9-slim
WORKDIR /model-client
COPY ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt
COPY . .
CMD ["python3", "client.py"]
