FROM python:3.9
WORKDIR /app
ENV HF_HOME="/app/cache/hf_cache_home"
# Copy requirements.txt to the docker image and install packages
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -i https://mirrors.cloud.tencent.com/pypi/simple -r requirements.txt
# Set the WORKDIR to be the folder
COPY . ./
# # Use gunicorn as the entrypoint
CMD python3 app.py
