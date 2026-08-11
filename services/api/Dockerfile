FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app
RUN addgroup --system app && adduser --system --ingroup app app

COPY pyproject.toml requirements.lock README.md ./
COPY src ./src
RUN python -m pip install --no-cache-dir -c requirements.lock .

COPY release ./release
COPY infra ./infra
COPY services ./services
USER app
EXPOSE 8000
CMD ["uvicorn", "services.api.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
