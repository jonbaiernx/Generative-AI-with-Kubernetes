FROM python:3.10-slim
WORKDIR /llmcompare
RUN apt-get update -y && apt-get install -y git-lfs && git lfs install
RUN apt-get install -y build-essential && apt-get install -y gcc-11 g++-11
COPY ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt
RUN mkdir Orca-2-7B-GGUF && \
    mkdir Mistral-7B-Instruct-v0.2-GGUF 
ENV HF_HUB_ENABLE_HF_TRANSFER=1
RUN huggingface-cli download TheBloke/Orca-2-7B-GGUF config.json --local-dir Orca-2-7B-GGUF --local-dir-use-symlinks False && \
    huggingface-cli download TheBloke/Orca-2-7B-GGUF USE_POLICY.md --local-dir Orca-2-7B-GGUF --local-dir-use-symlinks False && \
    huggingface-cli download TheBloke/Orca-2-7B-GGUF LICENSE.txt --local-dir Orca-2-7B-GGUF --local-dir-use-symlinks False && \
    huggingface-cli download TheBloke/Orca-2-7B-GGUF Notice --local-dir Orca-2-7B-GGUF --local-dir-use-symlinks False && \
    huggingface-cli download TheBloke/Orca-2-7B-GGUF README.md --local-dir Orca-2-7B-GGUF --local-dir-use-symlinks False  && \
    huggingface-cli download TheBloke/Orca-2-7B-GGUF orca-2-7b.Q4_K_M.gguf --local-dir Orca-2-7B-GGUF --local-dir-use-symlinks False
RUN huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.2-GGUF config.json --local-dir Mistral-7B-Instruct-v0.2-GGUF --local-dir-use-symlinks False && \
    huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.2-GGUF README.md --local-dir Mistral-7B-Instruct-v0.2-GGUF --local-dir-use-symlinks False && \
    huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.2-GGUF mistral-7b-instruct-v0.2.Q5_K_M.gguf --local-dir Mistral-7B-Instruct-v0.2-GGUF --local-dir-use-symlinks False
COPY . .
EXPOSE 7801
CMD ["gradio", "app.py"]