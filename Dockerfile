FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libass-dev \
    curl \
    gnupg \
    && curl https://sdk.cloud.google.com | bash \
    && rm -rf /var/lib/apt/lists/*

ENV PATH=$PATH:/root/google-cloud-sdk/bin

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1