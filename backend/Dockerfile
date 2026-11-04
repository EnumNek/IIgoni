# Образ бэкенда. Контекст сборки — корень репозитория (см. docker-compose.yml),
# в образ попадают только backend/ и файлы зависимостей ml/ и agents/ (Dockerfile.dockerignore).
FROM python:3.11-slim

# WITH_ML=1 — доустановить зависимости ml/ и agents/ (PyTorch Geometric, SUMO),
# чтобы бэкенд вызывал настоящие ml/predict.py и agents/evaluate.py, а не заглушки
ARG WITH_ML=0

ENV PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1
COPY . /src
RUN pip install --no-cache-dir -r /src/backend/requirements.txt \
 && if [ "$WITH_ML" = "1" ]; then \
      for f in /src/ml/requirements.txt /src/agents/requirements.txt; do \
        if [ -f "$f" ]; then pip install --no-cache-dir -r "$f"; fi; \
      done; \
    fi

WORKDIR /src/backend
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
