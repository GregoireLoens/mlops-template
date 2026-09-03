"""Arbitrage de réentraînement : drift + volume + performance (BP3, étape 5).

Règle (toutes conditions cumulatives, seuils dans `params.yaml/monitoring`) :

1. `dataset_drift is True` (part des colonnes dérivées > `share_threshold`)
   OU performance dégradée (`degraded is True` via la vérité retardée) ;
2. ET volume de nouvelles données suffisant (`n_current >= min_new_rows`) —
   on ne réentraîne jamais sur un échantillon non significatif ;
3. ET cooldown respecté (`cooldown_hours` depuis le dernier réentraînement
   enregistré dans `reports/monitoring/retrain_state.json`) — évite les
   boucles de retrain quotidiennes sur un drift persistant.

Sortie : `RetrainDecision(should_retrain, reasons)` — raisons lisibles,
journalisées. Le déclenchement effectif passe par Dagster (sensor
`monitoring_pipeline.py`) ou le CLI `make retrain-if-drifted`.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT, load_params

STATE_PATH = PROJECT_ROOT / "reports" / "monitoring" / "retrain_state.json"


@dataclass(frozen=True)
class RetrainDecision:
    should_retrain: bool
    reasons: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "should_retrain": self.should_retrain,
            "reasons": self.reasons,
            "summary": self.summary,
        }


def load_summary(summary_path: Path | None = None) -> dict[str, Any]:
    path = summary_path or (PROJECT_ROOT / "reports" / "monitoring" / "drift_summary.json")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _last_retrain_at() -> datetime | None:
    try:
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        ts = state.get("last_retrain_at")
        return datetime.fromisoformat(ts) if ts else None
    except (OSError, json.JSONDecodeError, ValueError):
        return None


def record_retrain() -> None:
    """Horodate un réentraînement déclenché (démarre le cooldown)."""
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(
        json.dumps({"last_retrain_at": datetime.now(UTC).isoformat()}, indent=2) + "\n",
        encoding="utf-8",
    )


def decide(summary: dict[str, Any] | None = None) -> RetrainDecision:
    """Applique la règle d'arbitrage au résumé du dernier drift-report."""
    params = load_params()
    summary = summary if summary is not None else load_summary()
    if not summary:
        return RetrainDecision(False, ["aucun drift_summary.json : lancer `make drift-report`"])

    reasons: list[str] = []
    drift = bool(summary.get("dataset_drift"))
    perf = summary.get("performance", {}) if isinstance(summary.get("performance"), dict) else {}
    degraded = bool(perf.get("degraded"))
    n_current = int(summary.get("n_current", 0))
    min_rows = params.monitoring.retrain.min_new_rows

    signal = drift or degraded
    if drift:
        drifted = summary.get("drifted_columns", [])
        reasons.append(f"drift détecté sur {drifted}")
    if degraded:
        reasons.append(
            f"performance dégradée (roc_auc={perf.get('roc_auc')}, drop={perf.get('roc_auc_drop')})"
        )
    if not signal:
        return RetrainDecision(False, ["ni drift ni dégradation : pas de réentraînement"])

    if n_current < min_rows:
        return RetrainDecision(
            False,
            reasons + [f"volume insuffisant ({n_current} < {min_rows} lignes)"],
            summary,
        )

    last = _last_retrain_at()
    cooldown = params.monitoring.retrain.cooldown_hours
    if last is not None:
        elapsed_h = (datetime.now(UTC) - last).total_seconds() / 3600
        if elapsed_h < cooldown:
            return RetrainDecision(
                False,
                reasons + [f"cooldown actif ({elapsed_h:.1f}h < {cooldown}h)"],
                summary,
            )

    return RetrainDecision(True, reasons + [f"volume OK ({n_current} >= {min_rows})"], summary)


def trigger_retrain() -> int:
    """Déclenche le pipeline de training (DVC repro) après décision positive.

    Retourne le code de sortie du training (0 = réentraînement lancé et
    réussi). Enregistre l'horodatage AVANT le run pour armer le cooldown
    même si le run échoue ensuite.
    """
    decision = decide()
    print(f"[retrain] should_retrain={decision.should_retrain} : {'; '.join(decision.reasons)}")
    if not decision.should_retrain:
        return 0
    record_retrain()
    result = subprocess.run(
        [sys.executable, "-m", "src.training.train"],
        cwd=PROJECT_ROOT,
        capture_output=False,
    )
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="affiche la décision sans déclencher")
    args = parser.parse_args()
    if args.check:
        decision = decide()
        print(json.dumps(decision.to_dict(), indent=2))
        sys.exit(0 if not decision.should_retrain else 2)
    sys.exit(trigger_retrain())


if __name__ == "__main__":
    main()
