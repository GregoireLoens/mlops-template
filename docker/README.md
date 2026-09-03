# docker/ — stack locale

| Service  | Rôle                                           | Commande           |
| -------- | ---------------------------------------------- | ------------------ |
| mlflow   | Tracking server + registre + artefacts (proxy) | `make up`          |
| postgres | Backend MLflow alternatif (profil `postgres`)  | `make up-postgres` |

## Points clés

- **Persistance** : volumes nommés `mlops-template_mlflow-data` (sqlite + artefacts) et `mlops-template_postgres-data`. `make down` conserve les données, `make down-volumes` remet à zéro.
- **Artefacts servis par le serveur** (`--serve-artifacts --artifacts-destination /mlflow/artifacts`) : les clients MLflow téléchargent les artefacts via HTTP (`mlflow-artifacts:/`), jamais via un montage disque. C'est la seule configuration qui reste valide quand on migre le stockage artefacts vers S3/MinIO.
- **Montée en charge (documentée, hors scope local)** : `--backend-store-uri postgresql://...`, `--artifacts-destination s3://bucket/prefix`, workers gunicorn devant. Le client ne change pas : même `MLFLOW_TRACKING_URI`, mêmes APIs registre.
- La configuration passe par `.env` (voir `../.env.example`) : port, URI backend, secrets — jamais en dur.
