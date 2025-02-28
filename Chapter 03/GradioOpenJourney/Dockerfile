FROM python:3.9-slim
WORKDIR /oj_app
COPY ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt
COPY . .
EXPOSE 7805
CMD ["gradio", "app.py"]