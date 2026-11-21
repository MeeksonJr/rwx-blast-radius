FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SKIP_INTEGRATION=0

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Clone pinned sqlglot commit eb11aec4be8cb2122afe713a4816662c3f0fb2fa
RUN git init /sqlglot \
    && cd /sqlglot \
    && git remote add origin https://github.com/tobymao/sqlglot.git \
    && git fetch --depth 1 origin eb11aec4be8cb2122afe713a4816662c3f0fb2fa \
    && git checkout FETCH_HEAD \
    && rm -rf /sqlglot/.git

# Install pinned test runtime dependencies from requirements.lock
RUN pip install --no-cache-dir \
    duckdb==2.0.0 \
    mypy_extensions==1.2.0 \
    numpy==2.5.5 \
    pandas==3.1.1 \
    python-dateutil==4.0.0 \
    pytz==2026.6 \
    setuptools==84.2.1 \
    typing_extensions==4.16.0 \
    tzdata==2026.6

WORKDIR /app
COPY super_index.json.gz /app/super_index.json.gz
COPY selector.py /app/selector.py

RUN chmod -R a+rX /app /sqlglot

CMD ["python", "/app/selector.py"]
