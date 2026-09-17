# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Copy dio-serve package files and install
COPY dio-serve/pyproject.toml dio-serve/README.md ./
COPY dio-serve/src ./src
RUN pip install --no-cache-dir .

# Non-root user for security
RUN useradd --create-home --shell /usr/sbin/nologin dio \
    && chown -R dio:dio /app
USER dio

ENTRYPOINT ["dio", "mcp"]
