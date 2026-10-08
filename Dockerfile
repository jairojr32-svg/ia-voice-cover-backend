# Use a base Python 3.10, que é a mais estável para RVC
FROM python:3.10-slim

# Instala dependências essenciais do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Define o diretório de trabalho
WORKDIR /app

# Copia e instala as dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o resto do código da aplicação
COPY . .

# Roda o script para baixar os modelos de voz durante a construção
RUN python download_popular_models.py

# Expõe a porta e define o comando de inicialização
EXPOSE 7860
CMD ["uvicorn", "server:py", "--host", "0.0.0.0", "--port", "7860"]
