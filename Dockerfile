FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/user/.cache/huggingface

# Run as a non-root user (also what Hugging Face Spaces expects)
RUN useradd -m -u 1000 user
WORKDIR /app

# Dependencies first so this layer is cached when only the code changes
COPY --chown=user requirements.txt .
RUN pip install -r requirements.txt

USER user

# Download the models at build time so container start-up does not have to
# fetch ~1 GB from the Hugging Face Hub every time.
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
AutoTokenizer.from_pretrained('google/flan-t5-base'); \
AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-base')"

COPY --chown=user . .

EXPOSE 8080

# Hosting platforms (Render, Railway, ...) inject $PORT; default to 8080.
# One worker: each worker would load its own copy of the models into RAM.
# Long timeout: the model loads inside the worker at start-up.
CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:${PORT:-8080} --workers 1 --threads 4 --timeout 300 app:app"]
