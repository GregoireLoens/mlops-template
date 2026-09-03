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
class DriftConfig:
    share_threshold: float
    pvalue_threshold: float
    min_current_rows: int
    min_reference_rows: int


@dataclass(frozen=True)
class PerformanceConfig:
    min_roc_auc: float
    min_accuracy: float
    drop_tolerance: float


@dataclass(frozen=True)
class RetrainConfig:
    min_new_rows: int
    cooldown_hours: float


@dataclass(frozen=True)
class MonitoringConfig:
    reference_path: str
    inferences_dir: str
    reports_dir: str
    drift: DriftConfig
    performance: PerformanceConfig
    retrain: RetrainConfig


@dataclass(frozen=True)
class Params:
    data: DataConfig
    features: FeaturesConfig
    train: TrainConfig
    eval: EvalConfig
    monitoring: MonitoringConfig


def load_params(path: Path = PARAMS_FILE) -> Params:
    with open(path, encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)
    return Params(
        data=DataConfig(**raw["data"]),
        features=FeaturesConfig(**raw["features"]),
        train=TrainConfig(**raw["train"]),
        eval=EvalConfig(**raw["eval"]),
        monitoring=_parse_monitoring(raw.get("monitoring", {})),
    )


def _parse_monitoring(raw: dict[str, Any]) -> MonitoringConfig:
    drift = DriftConfig(**raw.get("drift", {}))
    performance = PerformanceConfig(**raw.get("performance", {}))
    retrain = RetrainConfig(**raw.get("retrain", {}))
    return MonitoringConfig(
        reference_path=raw.get("reference_path", "data/prepared/train.csv"),
        inferences_dir=raw.get("inferences_dir", "data/inferences"),
        reports_dir=raw.get("reports_dir", "reports/monitoring"),
        drift=drift,
        performance=performance,
        retrain=retrain,
    )
