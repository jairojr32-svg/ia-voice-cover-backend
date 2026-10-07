FROM python:3.11-slim

# ffmpeg do sistema (para prévias/decodificação) + libsndfile p/ soundfile + git + curl
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# HF Spaces roda como usuário não-root (uid 1000): cache do torch/demucs precisa ser gravável
ENV HOME=/app \
    TORCH_HOME=/app/.cache/torch \
    XDG_CACHE_HOME=/app/.cache \
    ALLOWED_ORIGINS=*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .
COPY rvc_infer.py .
COPY download_popular_models.py .

RUN mkdir -p /app/.cache /app/models && chmod -R 777 /app

EXPOSE 7860
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
