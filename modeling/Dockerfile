FROM huggingface/transformers-pytorch-cpu
WORKDIR /app
# Copy requirements.txt to the docker image and install packages
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -i https://mirrors.cloud.tencent.com/pypi/simple -r requirements.txt
CMD jupyter lab --ip 0.0.0.0 --allow-root --no-browser
