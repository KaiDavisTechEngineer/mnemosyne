FROM python:3.12-slim AS builder
WORKDIR /build
RUN pip install --no-cache-dir --upgrade pip build
COPY pyproject.toml ./
COPY mnemosyne/ ./mnemosyne/
RUN python -m build --wheel --outdir /wheels

FROM python:3.12-slim AS runtime
LABEL org.opencontainers.image.title="MNEMOSYNE"
LABEL org.opencontainers.image.description="A society of causally-self-modeling agents"
LABEL org.opencontainers.image.licenses="MIT"
COPY --from=builder /wheels/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm /tmp/*.whl
WORKDIR /work
ENTRYPOINT ["mnemosyne"]
CMD ["--help"]
