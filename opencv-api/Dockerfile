# ── Gabarito Reader API — Dockerfile ──────────────────────────────────────────
FROM python:3.11-slim

# Dependências do sistema para OpenCV (libGL, libgthread, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /api

# Instala dependências Python primeiro (camada cacheada)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o código
COPY app/ ./app/

# Usuário não-root (boa prática de segurança)
RUN useradd -m apiuser
USER apiuser

EXPOSE 8000

# Inicia com uvicorn; ajuste workers conforme CPUs disponíveis
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
