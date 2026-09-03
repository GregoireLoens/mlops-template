"""Logging structuré des inférences servies (BP3, étape 1).

Chaque appel `/predict` réussi persiste UNE ligne JSONL :
horodatage ISO 8601, prediction_id (uuid4, clé de jointure avec la vérité
terrain retardée), version du modèle servi, features reçues, prédiction et
probabilité associées.

Stockage local partitionné par jour (`data/inferences/YYYY-MM-DD/`) :
format JSONL append-only (un crash ne corrompt jamais les lignes déjà
écrites), convertible en Parquet à la lecture. L'écriture est appelée via
FastAPI `BackgroundTasks` : elle ne bloque jamais la réponse `/predict`.

Activation : `LOG_INFERENCES=true/false` (défaut true), répertoire
surchargable par `INFERENCES_DIR` (tests, conteneurs).
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT

# Verrou inter-threads : uvicorn workers-threads partagent le processus, les
# BackgroundTasks s'exécutent dans le même processus — l'append reste atomique.
_LOCK = threading.Lock()

ENV_ENABLED = "LOG_INFERENCES"
ENV_DIR = "INFERENCES_DIR"


def logging_enabled() -> bool:
    """Interrupteur `LOG_INFERENCES` (défaut true, insensible à la casse)."""
    return os.getenv(ENV_ENABLED, "true").strip().lower() not in {"0", "false", "no", "off"}


def inferences_dir() -> Path:
    override = os.getenv(ENV_DIR)
    if override:
        return Path(override)
    return PROJECT_ROOT / "data" / "inferences"


def new_prediction_id() -> str:
    """Identifiant unique de requête (clé de jointure avec la vérité retardée)."""
    return uuid.uuid4().hex


def build_record(
    features: dict[str, Any],
    churn_probability: float,
    churn: bool,
    model_version: str | None,
    prediction_id: str | None = None,
) -> dict[str, Any]:
    """Construit une ligne de log typée (sérialisable JSON)."""
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "prediction_id": prediction_id or new_prediction_id(),
        "model_version": model_version,
        "features": features,
        "churn_probability": float(churn_probability),
        "churn": bool(churn),
    }


def append_record(record: dict[str, Any], base_dir: Path | None = None) -> Path:
    """Append synchrone d'une ligne JSONL dans la partition du jour.

    Retourne le chemin du fichier (utile aux tests). No-op si le logging
    est désactivé — le predicteur reste le seul juge via `logging_enabled`.
    """
    base = base_dir or inferences_dir()
    day = datetime.now(UTC).strftime("%Y-%m-%d")
    path = base / day / "inferences.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, default=str)
    with _LOCK, open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    return path


def log_inference(
    features: dict[str, Any],
    churn_probability: float,
    churn: bool,
    model_version: str | None,
    prediction_id: str | None = None,
) -> Path | None:
    """Point d'entrée du serving : construit + persiste, sauf si désactivé."""
    if not logging_enabled():
        return None
    record = build_record(
        features, churn_probability, churn, model_version, prediction_id=prediction_id
    )
    return append_record(record)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Lit un fichier d'inférences (une erreur de parsing = ligne ignorée)."""
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def latest_inference_frames(
    base_dir: Path | None = None, limit: int = 10_000
) -> list[dict[str, Any]]:
    """Dernières lignes de logs, partitions les plus récentes d'abord."""
    base = base_dir or inferences_dir()
    if not base.exists():
        return []
    files = sorted(base.glob("*/inferences.jsonl"), reverse=True)
    out: list[dict[str, Any]] = []
    for path in files:
        for record in reversed(read_jsonl(path)):
            out.append(record)
            if len(out) >= limit:
                return out
    return out
