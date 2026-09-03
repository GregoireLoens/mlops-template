"""Chargement du modèle challengé : registre MLflow PAR ALIAS, fallback local.

Principe : tests modèle et promotion chargent le modèle via l'alias
`challenger` — jamais par un chemin disque. Le fallback local (models/)
sert uniquement aux tests unitaires sans serveur (CI rapides).
"""

from __future__ import annotations

import os

import joblib
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.pipeline import Pipeline
from src.config import Params, load_params
from src.training.train import MODEL_PATH


def load_challenger(params: Params | None = None) -> tuple[Pipeline, str]:
    """Retourne (modèle, source) : 'registry v<N>' ou 'local' (fallback)."""
    params = params or load_params()
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if uri:
        mlflow.set_tracking_uri(uri)
    try:
        client = MlflowClient()
        mv = client.get_model_version_by_alias(params.train.model_name, "challenger")
        # Syntaxe alias : models:/name@alias (`/stage` désigne les stages).
        model = mlflow.sklearn.load_model(f"models:/{params.train.model_name}@challenger")
        return model, f"registry v{mv.version}"
    except Exception:
        # Pas de serveur / pas d'alias : fallback disque pour les tests unitaires.
        return joblib.load(MODEL_PATH), "local"
