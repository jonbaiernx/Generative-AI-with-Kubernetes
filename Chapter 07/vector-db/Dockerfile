FROM python:3.9-slim
WORKDIR /vector-db
COPY ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt
COPY . .
EXPOSE 7800
CMD ["gradio", "vector-db.py"]
