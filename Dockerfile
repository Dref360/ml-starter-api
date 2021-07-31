FROM python:3.7.11-slim
ENV PYTHONFAULTHANDLER=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONHASHSEED=random \
  PIP_NO_CACHE_DIR=off \
  PIP_DISABLE_PIP_VERSION_CHECK=on \
  PIP_DEFAULT_TIMEOUT=100 \
  POETRY_VERSION=1.1.7 \
  POETRY_HOME="/usr/local/poetry"

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y curl && \
    rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py | python \
    && ln -sf /usr/local/poetry/bin/poetry /usr/local/bin/poetry

# Install dependencies.
COPY poetry.lock pyproject.toml /app/

WORKDIR /app
RUN poetry config virtualenvs.create false && \
 poetry install --no-interaction --no-ansi --no-root --no-dev

# Install the project.
COPY . /app/
RUN poetry install --no-interaction --no-ansi --no-dev
ENV PORT=8080
CMD ["sh","-c","umask 0002; poetry run python runner.py --port ${PORT}"]
