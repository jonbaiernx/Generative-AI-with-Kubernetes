FROM python:3.10-slim
WORKDIR /llm-rag
RUN apt-get update -y && apt-get install -y git-lfs && git lfs install
RUN apt-get install -y build-essential && apt-get install -y gcc-11 g++-11
COPY ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt
RUN mkdir phi-2-GGUF 
ENV HF_HUB_ENABLE_HF_TRANSFER=1
RUN huggingface-cli download TheBloke/phi-2-GGUF phi-2.Q4_K_M.gguf --local-dir phi-2-GGUF --local-dir-use-symlinks False && \
    huggingface-cli download TheBloke/phi-2-GGUF config.json --local-dir phi-2-GGUF --local-dir-use-symlinks False && \
    huggingface-cli download TheBloke/phi-2-GGUF README.md --local-dir phi-2-GGUF --local-dir-use-symlinks False  
COPY . .
EXPOSE 7801
CMD ["gradio", "app.py"]