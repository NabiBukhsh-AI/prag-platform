# Multi-stage, so the runtime image carries no build toolchain. A smaller image is a smaller
# attack surface as much as a faster pull.
FROM python:3.12-slim AS builder

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir --upgrade pip build \
 && pip wheel --no-cache-dir --wheel-dir /wheels ".[api]"

FROM python:3.12-slim

# A non-root user by default. Nothing here needs root, and a container that runs as root only
# because nobody changed the default is a container that will run as root in production.
RUN useradd --create-home --uid 10001 prag
WORKDIR /app

COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels prag[api] \
 && rm -rf /wheels

COPY scripts ./scripts

USER prag
EXPOSE 8080

# One worker. Horizontal scaling is by replica rather than by in-container workers, so that a
# replica is the unit of both scheduling and failure and the two never disagree.
CMD ["uvicorn", "prag.api.http.main:app", "--host", "0.0.0.0", "--port", "8080"]
