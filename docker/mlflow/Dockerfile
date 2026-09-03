# Image MLflow du template : l'officielle + les drivers des backends alternatifs.
# Version épinglée pour la reproductibilité (serveur et client sur la même majeure).
ARG MLFLOW_VERSION=3.8.1
FROM ghcr.io/mlflow/mlflow:v${MLFLOW_VERSION}

# psycopg2 : backend postgres (profil compose "postgres")
# boto3     : artefacts/tracking sur S3 ou S3-compatible (MinIO) — mode scalable
RUN pip install --no-cache-dir \
      "psycopg2-binary>=2.9,<3" \
      "boto3>=1.35,<2"

EXPOSE 5000
