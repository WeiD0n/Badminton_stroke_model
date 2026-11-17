FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxcb1 \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

COPY web_app/backend/requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

COPY web_app/backend /app/web_app/backend
COPY training /app/training
COPY models/form_evaluation /app/models/form_evaluation
COPY evaluation/700_per_class_full_features/run_059 /app/evaluation/700_per_class_full_features/run_059

EXPOSE 8080

CMD ["uvicorn", "web_app.backend.main:app", "--host", "0.0.0.0", "--port", "8080"]
