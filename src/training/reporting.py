"""Artefacts de reporting du training : model card + matrice de confusion.

Le model card est minimal mais versionné avec le modèle (out DVC models/ et
artefact MLflow) : qui a entraîné quoi, avec quelles données, quelles
métriques, quels seuils de gate.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import matplotlib

matplotlib.use("Agg")  # pas d'affichage : pipeline headless (DVC/Dagster/CI)
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from src.config import PROJECT_ROOT, Params

MODEL_CARD_PATH = PROJECT_ROOT / "models" / "model_card.md"
CONFUSION_PATH = PROJECT_ROOT / "models" / "confusion_matrix.png"


def plot_confusion_matrix(model: Pipeline, test_df: pd.DataFrame, params: Params) -> None:
    X = test_df[params.features.numeric + params.features.categorical]
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay.from_predictions(
        test_df[params.data.target],
        model.predict(X),
        ax=ax,
        colorbar=False,
    )
    ax.set_title(f"churn — {params.train.model_type}")
    fig.tight_layout()
    fig.savefig(CONFUSION_PATH, dpi=120)
    plt.close(fig)


def build_model_card(
    params: Params, metrics: dict[str, float | str], hyperparams: dict[str, Any]
) -> str:
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    raw = pd.read_csv(PROJECT_ROOT / params.data.raw_path, usecols=["signup_ts"])
    gate = (
        f"accuracy>={params.eval.min_accuracy}, f1>={params.eval.min_f1}, "
        f"roc_auc>={params.eval.min_roc_auc}"
    )
    return f"""# Model card — {params.train.model_name}

Générée automatiquement par l'étape `train` ({now}).

| Champ | Valeur |
| --- | --- |
| Modèle | {params.train.model_type} (`{hyperparams}`) |
| Cible | {params.data.target} |
| Features numériques | {params.features.numeric} |
| Features catégorielles | {params.features.categorical} |
| Split | test_size={params.data.test_size}, seed={params.data.seed} |
| Fenêtre de données | {raw["signup_ts"].min()} -> {raw["signup_ts"].max()} |
| Métriques (test) | {metrics} |
| Gate d'évaluation | {gate} |

## Usage

Le modèle est servi par alias de registre (jamais par chemin disque) :
`challenger` après entraînement, `prod` après promotion (gate = tests modèle
verts + métriques supérieures). Préprocessing inclus dans le Pipeline sklearn.

## Limites

Données **simulées** (template) : les seuils et la fenêtre temporelle sont
à recalibrer sur les données réelles du client. Aucune donnée personnelle.
"""
