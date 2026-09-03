"""Définitions Feast du template : entité, source, feature view.

Fichier auto-portant (importable par le CLI `feast apply` avec cwd=features/
ET depuis la racine du repo via `features.feature_views`) : Feast exécute
les fichiers du repo sans contexte de package, d'où le sys.path explicite.

Règle du template : le feature view expose EXACTEMENT les features du
training DVC (params.yaml). L'équivalence des deux chemins est testée —
sinon skew train/serving.
"""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float64, Int64, String

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import Params, load_params  # noqa: E402

# value_type explicite : obligatoire dans la prochaine release Feast.
customer = Entity(name="customer", join_keys=["customer_id"], value_type=ValueType.STRING)

# Colonnes entières côté parquet ; les autres numériques sont des Float64,
# les catégorielles des String. Le mapping suit les dtypes de prepare.py.
INT64_COLUMNS = {"age", "num_support_calls", "has_premium"}


def _fields(params: Params) -> list[Field]:
    dtypes: dict[str, object] = {}
    for col in params.features.numeric:
        dtypes[col] = Int64 if col in INT64_COLUMNS else Float64
    for col in params.features.categorical:
        dtypes[col] = String
    return [Field(name=name, dtype=dtype) for name, dtype in dtypes.items()]  # type: ignore[arg-type]


def build_customer_profile(source_path: str | None = None) -> FeatureView:
    """Feature view client, paramétrable pour les tests (repo temporaire)."""
    params = load_params()
    source = FileSource(
        name="customer_prepared",
        path=source_path or str(ROOT / params.data.train_path).replace(".csv", ".parquet"),
        timestamp_field="signup_ts",
    )
    return FeatureView(
        name="customer_profile",
        entities=[customer],
        ttl=timedelta(days=400),  # > fenêtre de données : aucun expiring en local
        schema=_fields(params),
        source=source,
    )


customer_profile = build_customer_profile()
