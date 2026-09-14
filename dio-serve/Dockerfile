# syntax=docker/dockerfile:1

# DIO - Distributed Inference Orchestrator (gateway image)
#   Docs: https://github.com/nisaral/DIO
#
# Build from the package directory (context must be dio-serve/):
#   docker build -t dio-serve:0.4.0 .
#
# Run behind the bundled Ollama engine (see docker-compose.yml for the easy path):
#   docker run --rm -p 8085:8085 \
#     -v "$PWD/docker/dio.yaml:/app/dio.yaml:ro" dio-serve:0.4.0
#
# SECURITY - DIO SHIPS NO AUTHENTICATION.
# The shipped config binds 0.0.0.0 (required so the container's published port
# is reachable) and the /debug/* routes (stats, chaos, backend dumps) are
# unauthenticated. Always place this container behind an authenticating reverse
# proxy (mTLS / OAuth2 / API-key gateway) and never expose 8085 to an untrusted
# network directly.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DIO_HOST=0.0.0.0 \
    DIO_PORT=8085

WORKDIR /app

# Install runtime deps + the package itself from the build context.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

# Container-oriented default config (ollama backend -> http://ollama:11434).
# Override at runtime by mounting your own dio.yaml over /app/dio.yaml.
COPY docker/dio.yaml ./dio.yaml

# Reference config kept for `dio init`-style introspection / docs.
COPY dio.example.yaml ./dio.example.yaml

# Unprivileged runtime user that also owns /app (keeps a mounted dio.yaml readable).
RUN useradd --create-home --shell /usr/sbin/nologin dio \
    && chown -R dio:dio /app
USER dio

EXPOSE 8085

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8085/health', timeout=3).status == 200 else 1)"]

ENTRYPOINT ["dio"]
CMD ["serve"]
