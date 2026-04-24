FROM python:3.10-slim

# ffmpeg for video processing, libass-dev for SRT subtitle burn-in
# ImageMagick removed — no longer needed since we replaced TextClip with SRT
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libass-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Single worker — render jobs are CPU/memory intensive.
# Let Cloud Run scale by spinning up new instances, not new workers.
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1
