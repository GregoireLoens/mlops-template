# Image de serving : project + FastAPI. Le modèle n'est PAS baked dans
# l'image : il est chargé au démarrage depuis le registre MLflow par alias
# (prod) — le déploiement reste découplé du training.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MLFLOW_TRACKING_URI=http://mlflow:5000 \
    MODEL_NAME=churn-template \
    MODEL_ALIAS=prod

WORKDIR /app

COPY pyproject.toml README.md LICENSE params.yaml ./
COPY src ./src
RUN pip install --no-cache-dir ".[serving]"

EXPOSE 8000
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
