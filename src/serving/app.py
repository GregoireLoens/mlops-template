"""API de serving du modèle : chargement PAR ALIAS du registre MLflow.

- /health  : statut + version du modèle servi (probe compose/k8s)
- /predict : features brutes -> probabilité de churn (Pipeline sklearn :
  préprocessing identique au training — zéro train/serving skew)
- /reload  : recharge le modèle pointé par l'alias (utilisé par le rollback)

Config par variables d'environnement (aucun chemin ni clé en dur) :
MODEL_NAME, MODEL_ALIAS (défaut prod), MLFLOW_TRACKING_URI,
SERVE_FAILURE=1 pour simuler un canary dégradé (démo rollback).
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Annotated, Literal

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, Field
from sklearn.pipeline import Pipeline
from src.config import load_params

# Catégories du dataset simulé (cf. generate_raw) : le OneHotEncoder entraîné
# les connaît ; toute valeur hors Literal est rejetée à la porte (422), on
# n'envoie jamais silencieusement des vecteurs vides au modèle.
ContractType = Literal["month_to_month", "one_year", "two_year"]
SignupChannel = Literal["web", "mobile", "partner"]

_state: dict[str, str | None] = {"model_version": None}


@lru_cache(maxsize=1)
def _model() -> Pipeline:
    """Charge UNE fois le pipeline sklearn (thread-safe via lru_cache)."""
    params = load_params()
    alias = os.getenv("MODEL_ALIAS", "prod")
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if uri:
        mlflow.set_tracking_uri(uri)
    mv = MlflowClient().get_model_version_by_alias(params.train.model_name, alias)
    model = mlflow.sklearn.load_model(f"models:/{params.train.model_name}@{alias}")
    _state["model_version"] = f"v{mv.version} ({alias})"
    return model


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    _model()
    yield


app = FastAPI(title="churn-serving", version="0.1.0", lifespan=lifespan)


class PredictRequest(BaseModel):
    age: Annotated[int, Field(ge=18, le=75)]
    tenure_months: Annotated[float, Field(ge=0, le=120)]
    monthly_fee: Annotated[float, Field(ge=0, le=200)]
    num_support_calls: Annotated[int, Field(ge=0, le=50)]
    has_premium: Literal[0, 1]
    contract_type: ContractType
    signup_channel: SignupChannel


class PredictResponse(BaseModel):
    churn_probability: float
    churn: bool
    model_version: str | None


@app.get("/health")
def health() -> dict:
    if _state["model_version"] is None:
        raise HTTPException(status_code=503, detail="modèle non chargé")
    return {"status": "ok", "model_version": _state["model_version"]}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if os.getenv("SERVE_FAILURE") == "1":
        # Simulation d'un canary dégradé : la gate smoke-test doit le détecter.
        raise HTTPException(status_code=500, detail="échec simulé (SERVE_FAILURE)")
    df = pd.DataFrame([req.model_dump()])
    proba = float(_model().predict_proba(df)[0, 1])
    return PredictResponse(
        churn_probability=proba,
        churn=proba >= 0.5,
        model_version=_state["model_version"],
    )


@app.post("/reload")
def reload_model() -> dict:
    _model.cache_clear()
    _model()
    return {"status": "rechargé", "model_version": _state["model_version"]}
