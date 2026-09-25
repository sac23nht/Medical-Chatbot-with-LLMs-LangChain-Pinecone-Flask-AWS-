FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Run as a non-root user
RUN useradd -m -u 1000 user
WORKDIR /app

# Dependencies first so this layer is cached when only the code changes
COPY --chown=user requirements.txt .
RUN pip install -r requirements.txt

USER user
COPY --chown=user . .

EXPOSE 8501

# Hosting platforms (Render, Railway, ...) inject $PORT; default to Streamlit's 8501.
# Embeddings and the LLM are hosted APIs, so the app itself is light.
CMD ["sh", "-c", "exec streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0"]
