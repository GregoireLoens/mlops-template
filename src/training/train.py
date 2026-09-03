"""Étape DVC `train` : fit, évaluation, écriture des artefacts.

Le modèle est un sklearn.Pipeline complet (préprocessing inclus) : c'est
l'objet packagé, servi tel quel par l'API (étape 10).

MLflow est branché sur ces mêmes fonctions à l'étape 5 (log params/metrics/
artefacts + registre) — le script DVC et les jobs Dagster ne dupliquent rien.
À cette étape, on reste DVC-only : `metrics.json` (métriques DVC, commité)
et `models/model.pkl`.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.config import PROJECT_ROOT, Params, load_params

MODEL_PATH = PROJECT_ROOT / "models" / "model.pkl"
METRICS_PATH = PROJECT_ROOT / "metrics.json"


def build_pipeline(params: Params) -> Pipeline:
    """Pipeline complet : préprocessing inclus dans le modèle packagé.

    OneHotEncoder(handle_unknown="ignore") : une catégorie jamais vue au
    training ne doit pas faire planter le serving, seulement dégrader la
    prédiction (comportement testé à l'étape 6).
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), params.features.numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), params.features.categorical),
        ],
        remainder="drop",
    )
    if params.train.model_type == "logreg":
        estimator: Any = LogisticRegression(**params.train.logreg)
    elif params.train.model_type == "rf":
        estimator = RandomForestClassifier(**params.train.rf)
    else:
        raise ValueError(f"model_type inconnu : {params.train.model_type!r} (logreg|rf)")
    return Pipeline([("preprocess", preprocessor), ("model", estimator)])


def _X(df: pd.DataFrame, params: Params) -> pd.DataFrame:
    return df[params.features.numeric + params.features.categorical]


def train_model(train_df: pd.DataFrame, params: Params) -> Pipeline:
    model = build_pipeline(params)
    model.fit(_X(train_df, params), train_df[params.data.target])
    return model


def evaluate(model: Pipeline, test_df: pd.DataFrame, params: Params) -> dict[str, float]:
    y = test_df[params.data.target]
    proba = model.predict_proba(_X(test_df, params))[:, 1]
    return {
        "accuracy": float(accuracy_score(y, model.predict(_X(test_df, params)))),
        "f1": float(f1_score(y, model.predict(_X(test_df, params)))),
        "roc_auc": float(roc_auc_score(y, proba)),
    }


def run(params: Params) -> dict[str, float | str]:
    train_df = pd.read_csv(PROJECT_ROOT / params.data.train_path)
    test_df = pd.read_csv(PROJECT_ROOT / params.data.test_path)

    model = train_model(train_df, params)
    metrics: dict[str, float | str] = {
        **evaluate(model, test_df, params),
        "model_type": params.train.model_type,
        "n_train": float(len(train_df)),
        "n_test": float(len(test_df)),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def run_from_feast(params: Params) -> dict[str, float | str]:
    """Chemin Feast : mêmes fonctions, features servies par le feature store.

    Démonstration du chemin offline Feast (point-in-time correct) : le
    modèle NE PASSE PAS par models/ ni metrics.json (le chemin DVC reste
    la source de vérité batch) — on mesure et on compare. L'équivalence
    features Feast <-> CSV est testée dans tests/data/test_feast.py.
    """
    from src.features.store import load_training_frame  # import tardif (feast)

    train_df = load_training_frame(params)
    test_df = pd.read_csv(PROJECT_ROOT / params.data.test_path)
    model = train_model(train_df, params)
    metrics: dict[str, float | str] = {
        **evaluate(model, test_df, params),
        "model_type": params.train.model_type,
        "source": "feast",
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=["files", "feast"],
        default="files",
        help="files = data/prepared (DVC), feast = get_historical_features",
    )
    args = parser.parse_args()
    params = load_params()
    metrics = run(params) if args.source == "files" else run_from_feast(params)
    print(f"train OK — {metrics}")


if __name__ == "__main__":
    main()
