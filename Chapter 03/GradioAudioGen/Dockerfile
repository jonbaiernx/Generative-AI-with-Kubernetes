FROM python:3.9-slim
WORKDIR /ag_app
COPY ./requirements.txt requirements.txt
RUN apt update
RUN pip install --no-cache-dir --upgrade -r requirements.txt
RUN apt install -y git 
RUN pip install -U git+https://git@github.com/facebookresearch/audiocraft#egg=audiocraft
RUN apt install -y ffmpeg
COPY . .
EXPOSE 7806
CMD ["gradio", "app.py"]