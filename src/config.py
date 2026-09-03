"""Configuration centrale du pipeline.

params.yaml est la source de vérité des hyperparamètres et des chemins :
lue par DVC (dépendances `params` des étapes), par Dagster (étape 7) et
par les tests. Le chargement est typé pour échouer tôt si un champ manque.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# Racine du repo, déduite de l'emplacement du module (src/config.py).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARAMS_FILE = PROJECT_ROOT / "params.yaml"


@dataclass(frozen=True)
class DataConfig:
    raw_path: str
    train_path: str
    test_path: str
    target: str
    test_size: float
    seed: int


@dataclass(frozen=True)
class FeaturesConfig:
    numeric: list[str]
    categorical: list[str]


@dataclass(frozen=True)
class TrainConfig:
    model_type: str
    logreg: dict[str, Any]
    rf: dict[str, Any]
    model_name: str


@dataclass(frozen=True)
class EvalConfig:
    min_accuracy: float
    min_f1: float
    min_roc_auc: float


@dataclass(frozen=True)
class Params:
    data: DataConfig
    features: FeaturesConfig
    train: TrainConfig
    eval: EvalConfig


def load_params(path: Path = PARAMS_FILE) -> Params:
    with open(path, encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)
    return Params(
        data=DataConfig(**raw["data"]),
        features=FeaturesConfig(**raw["features"]),
        train=TrainConfig(**raw["train"]),
        eval=EvalConfig(**raw["eval"]),
    )
