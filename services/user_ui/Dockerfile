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

COPY services/user_ui/app.py ./services/user_ui/app.py
USER app
EXPOSE 8501
CMD ["streamlit", "run", "services/user_ui/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
