FROM python:3.10-slim
WORKDIR /mlflow-custom-metrics
RUN apt-get update -y && apt-get install -y git-lfs && git lfs install
RUN apt-get install -y build-essential && apt-get install -y gcc-11 g++-11
COPY ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt
ENV HF_HUB_ENABLE_HF_TRANSFER=1
RUN huggingface-cli download gpt2
RUN huggingface-cli download MBZUAI/LaMini-Flan-T5-77M
COPY . .
CMD model-custom-metrics-runs.py