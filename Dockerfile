FROM python:3.11.16-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        g++ \
        gcc \
        libc6-dev \
        libopenblas-dev \
        liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.12.16 /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PYTHONHASHSEED=0 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    BLIS_NUM_THREADS=1 \
    PYTENSOR_FLAGS="blas__ldflags=-lopenblas"

ENV PATH="/opt/venv/bin:${PATH}"

WORKDIR /workspace

COPY pyproject.toml uv.lock ./

RUN uv sync \
    --frozen \
    --all-groups \
    --no-install-project

CMD ["sleep", "infinity"]