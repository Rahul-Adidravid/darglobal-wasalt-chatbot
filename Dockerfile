FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FASTEMBED_CACHE_PATH=/app/models \
    PORT=7860

# Hugging Face Spaces runs containers as uid 1000
RUN useradd -m -u 1000 appuser
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY data ./data

# Download the small ONNX embedding model; build the vector index only if a prebuilt one
# was not shipped in data/index (building takes ~20 min on CPU, so it is normally prebuilt).
RUN python -c "from fastembed import TextEmbedding; from app.config import EMBED_MODEL, EMBED_CACHE; TextEmbedding(EMBED_MODEL, cache_dir=EMBED_CACHE)" \
 && (test -f data/index/embeddings.npy || python -m app.ingest) \
 && chown -R appuser:appuser /app

USER appuser
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s CMD python -c "import urllib.request,os;urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"7860\")}/health')"
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
