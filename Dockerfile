FROM python:3.11.16-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.12.16 /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_LINK_MODE=copy
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

WORKDIR /workspace

COPY pyproject.toml uv.lock ./

RUN uv sync \
    --frozen \
    --all-groups \
    --no-install-project

CMD ["sleep", "infinity"]