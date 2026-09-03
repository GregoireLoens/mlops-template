"""Métriques Prometheus du serving (BP3, étape 4).

Instrumente FastAPI avec le client officiel `prometheus_client` (déjà
dépendance transitive, ajoutée à l'extra `serving`) :

- `http_requests_total{method,path,status,model_version}` : compteur requêtes ;
- `inference_latency_seconds{model_version}` : histogramme latences /predict ;
- `data_drift_share` : gauge du dernier score de drift (rafraîchie à chaque
  scrape depuis `reports/monitoring/drift_summary.json` si présent) ;
- `prediction_drift_share` : gauge de la dérive de prédiction.

`/metrics` est servi par l'API mais NON exposé via nginx (compose : seul
:8001/:8002 en interne, l'edge :8090 ne route que `/predict` et `/health`).
Voir `docker/compose.yaml` (profil `monitoring`, Prometheus minimaliste).
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from src.config import PROJECT_ROOT

REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Requêtes HTTP servies.",
    ["method", "path", "status", "model_version"],
)
INFERENCE_LATENCY = Histogram(
    "inference_latency_seconds",
    "Latence d'inférence /predict (secondes).",
    ["model_version"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)
DATA_DRIFT_SHARE = Gauge(
    "data_drift_share",
    "Part des colonnes en dérive (dernier rapport Evidently, 0 si absent).",
)
PREDICTION_DRIFT_SHARE = Gauge(
    "prediction_drift_share",
    "Dérive de la prédiction : 1 si driftée, 0 sinon (dernier rapport).",
)
ERROR_RATE = Gauge(
    "serving_error_rate",
    "Taux d'erreur 5xx estimé (fenêtre glissante via compteurs).",
)

_ERRORS = 0
_TOTAL = 0


def observe_request(
    method: str, path: str, status: int, model_version: str | None, latency: float
) -> None:
    """Incrémente les compteurs après chaque réponse (jamais bloquant)."""
    global _ERRORS, _TOTAL
    version = model_version or "unknown"
    REQUESTS_TOTAL.labels(method, path, str(status), version).inc()
    if path == "/predict":
        INFERENCE_LATENCY.labels(version).observe(latency)
    if path == "/predict":
        _TOTAL += 1
        if status >= 500:
            _ERRORS += 1
        ERROR_RATE.set(_ERRORS / _TOTAL if _TOTAL else 0.0)


@contextmanager
def timed() -> Iterator[list[float]]:
    """Mesure la latence d'un bloc (début/fin monotones)."""
    start = time.monotonic()
    holder: list[float] = [0.0]
    try:
        yield holder
    finally:
        holder[0] = time.monotonic() - start


def refresh_drift_gauges(summary_path: Path | None = None) -> dict[str, Any]:
    """Relit le résumé JSON du dernier drift-report et met à jour les gauges.

    Retourne le résumé (dict vide si absent/illisible) — `/metrics` reste
    vert même sans rapport généré.
    """
    path = summary_path or (PROJECT_ROOT / "reports" / "monitoring" / "drift_summary.json")
    try:
        summary: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        DATA_DRIFT_SHARE.set(0.0)
        PREDICTION_DRIFT_SHARE.set(0.0)
        return {}
    DATA_DRIFT_SHARE.set(float(summary.get("drift_share", 0.0)))
    PREDICTION_DRIFT_SHARE.set(1.0 if summary.get("prediction_drift") else 0.0)
    return summary


def exposition() -> tuple[bytes, str]:
    """Payload `/metrics` : rafraîchit les gauges puis expose tout."""
    refresh_drift_gauges()
    return generate_latest(), CONTENT_TYPE_LATEST
