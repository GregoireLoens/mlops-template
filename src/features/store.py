"""Accès au feature store Feast (repo local file).

Place de Feast dans le template : couche de SERVING de features.
- chemin batch  : le training DVC lit data/prepared (source de vérité) ;
- chemin Feast  : `load_training_frame` (point-in-time correct, offline)
  et `get_online_features` pour le serving temps réel (étape 10).

Les deux chemins exposent les mêmes features — l'équivalence est testée
(tests/data/test_feast.py). Voir README racine : « DVC vs Feast ».
"""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
from feast import FeatureStore
from src.config import PROJECT_ROOT, Params

FEAST_REPO = PROJECT_ROOT / "features"


def get_store() -> FeatureStore:
    return FeatureStore(repo_path=str(FEAST_REPO))


def feature_references(params: Params) -> list[str]:
    return [
        f"customer_profile:{col}" for col in params.features.numeric + params.features.categorical
    ]


def materialize_latest(store: FeatureStore | None = None) -> None:
    """Materialize de la fenêtre complète du dataset vers l'online store.

    Idempotent : re-matérialiser la même fenêtre réécrit les mêmes valeurs.
    """
    store = store or get_store()
    store.materialize(
        start_date=datetime(2024, 12, 1, tzinfo=UTC),
        end_date=datetime.now(UTC),
    )


def historical_features_df(
    params: Params, store: FeatureStore, entity_df: pd.DataFrame
) -> pd.DataFrame:
    """Features point-in-time pour un entity_df donné (offline retrieval)."""
    retrieved = store.get_historical_features(
        entity_df=entity_df, features=feature_references(params)
    )
    return retrieved.to_df()


def load_training_frame(params: Params, store: FeatureStore | None = None) -> pd.DataFrame:
    """Training set via get_historical_features (point-in-time join correct).

    Retourne les mêmes colonnes de features que data/prepared/train.csv, la
    cible étant rattachée par clé (elle n'appartient pas au feature view).
    """
    store = store or get_store()
    train = pd.read_csv(PROJECT_ROOT / params.data.train_path, parse_dates=["signup_ts"])
    df = historical_features_df(params, store, train[["customer_id", "signup_ts"]])
    return df.merge(train[["customer_id", params.data.target]], on="customer_id")


if __name__ == "__main__":
    materialize_latest()
    print(f"online store matérialisé : {PROJECT_ROOT / 'data/feast/online_store.db'}")
