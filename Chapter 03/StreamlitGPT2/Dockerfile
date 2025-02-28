FROM python:3.9-slim
WORKDIR /streamlitGPT2
COPY ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit","run", "app.py","--server.port","8501","--server.enableWebsocketCompression","false"]

