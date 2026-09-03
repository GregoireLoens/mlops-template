"""Tests de métriques : seuils minimaux (params.yaml `eval`) sur le jeu de test.

C'est la première gate de promotion (étape 8) : un challenger sous les
seuils est refusé avant toute comparaison avec le champion.
"""

from __future__ import annotations

import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline


def _metrics(
    model: Pipeline, test_df: pd.DataFrame, features: list[str], target: str
) -> dict[str, float]:
    y = test_df[target]
    proba = model.predict_proba(test_df[features])[:, 1]
    return {
        "accuracy": float(accuracy_score(y, model.predict(test_df[features]))),
        "f1": float(f1_score(y, model.predict(test_df[features]))),
        "roc_auc": float(roc_auc_score(y, proba)),
    }


@pytest.fixture(scope="module")
def metrics(model, test_df, features, params) -> dict[str, float]:
    return _metrics(model, test_df, features, params.data.target)


def test_accuracy_au_dessus_du_seuil(metrics: dict[str, float], params) -> None:
    assert metrics["accuracy"] >= params.eval.min_accuracy


def test_f1_au_dessus_du_seuil(metrics: dict[str, float], params) -> None:
    assert metrics["f1"] >= params.eval.min_f1


def test_roc_auc_au_dessus_du_seuil(metrics: dict[str, float], params) -> None:
    assert metrics["roc_auc"] >= params.eval.min_roc_auc


def test_les_seuils_sont_coherents(params) -> None:
    # Garde-fou config : des seuils >= 1 rendraient toute promotion impossible.
    for seuil in (params.eval.min_accuracy, params.eval.min_f1, params.eval.min_roc_auc):
        assert 0.0 < seuil < 1.0
