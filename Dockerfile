# ── base image ───────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# TensorFlow & OpenCV native libs
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1     \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── python deps ──────────────────────────────────────────────────
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── code & model ─────────────────────────────────────────────────
COPY . .

# ── unprivileged user ────────────────────────────────────────────
RUN useradd -r -u 1001 -g root appuser
USER appuser

# gunicorn listens on 8000 inside the container
EXPOSE 8000
ENV PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2

# ── entrypoint ───────────────────────────────────────────────────
CMD ["gunicorn", "-c", "compose/gunicorn.conf.py", "app:app"]
