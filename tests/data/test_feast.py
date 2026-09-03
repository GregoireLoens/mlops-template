"""Tests Feast : repo temporaire, historical features, équivalence CSV/Feast.

Les tests montent un repo Feast ISOLÉ (tmp_path, source parquet temporaire) :
aucune dépendance aux sorties DVC du repo, aucun écrit dans features/.
L'équivalence entre le chemin Feast et le chemin CSV est le point clé :
c'est elle qui garantit l'absence de skew train/serving.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest
from feast import FeatureStore
from src.config import load_params
from src.data.generate_raw import generate
from src.data.prepare import split
from src.features.store import feature_references, historical_features_df


def _build_tmp_store(tmp_path: Path, train_df: pd.DataFrame) -> FeatureStore:
    """Repo Feast jetable : parquet source + config + apply des définitions."""
    parquet = tmp_path / "train.parquet"
    train_df.to_parquet(parquet, index=False)

    repo = tmp_path / "features"
    repo.mkdir()
    (repo / "feature_store.yaml").write_text(
        "project: mlops_template_test\n"
        "provider: local\n"
        f"registry: {tmp_path / 'registry.db'}\n"
        "online_store:\n  type: sqlite\n"
        f"  path: {tmp_path / 'online.db'}\n"
        "entity_key_serialization_version: 2\n"
    )
    store = FeatureStore(repo_path=str(repo))

    from features.feature_views import build_customer_profile, customer

    store.apply([customer, build_customer_profile(source_path=str(parquet))])
    return store


@pytest.fixture()
def tmp_train() -> pd.DataFrame:
    params = load_params()
    raw = generate(800, seed=5)
    train_df, _ = split(raw, params.data.target, params.data.test_size, params.data.seed)
    return train_df


def test_le_training_set_feast_est_identique_au_chemin_csv(
    tmp_path: Path, tmp_train: pd.DataFrame
) -> None:
    params = load_params()
    store = _build_tmp_store(tmp_path, tmp_train)

    feast_df = historical_features_df(params, store, tmp_train[["customer_id", "signup_ts"]]).merge(
        tmp_train[["customer_id", params.data.target]], on="customer_id"
    )

    cols = params.features.numeric + params.features.categorical + [params.data.target]
    from_csv = tmp_train.sort_values("customer_id")[cols].reset_index(drop=True)
    from_feast = feast_df.sort_values("customer_id")[cols].reset_index(drop=True)
    # check_dtype=False : CSV re-convertit en int64/float64 selon les nulls ;
    # les VALEURS doivent être strictement identiques.
    pd.testing.assert_frame_equal(from_csv, from_feast, check_dtype=False)


def test_le_materialize_sert_les_features_en_ligne(tmp_path: Path, tmp_train: pd.DataFrame) -> None:
    params = load_params()
    store = _build_tmp_store(tmp_path, tmp_train)

    store.materialize(
        start_date=datetime(2024, 12, 1, tzinfo=UTC),
        end_date=datetime.now(UTC),
    )
    customer_id = tmp_train["customer_id"].iloc[0]
    row = store.get_online_features(
        features=feature_references(params),
        entity_rows=[{"customer_id": customer_id}],
    ).to_dict()

    # Toutes les features du feature view sont servies, non nulles.
    for ref in feature_references(params):
        col = ref.split(":")[1]
        assert row[col][0] is not None, f"feature non servie : {col}"
