"""Pipeline Dagster de monitoring (BP3, étape 5) : drift périodique -> retrain.

Chaîne : `drifted` (Evidently sur la fenêtre d'inférences) -> `evaluated`
(arbitrage retrain_policy : drift + volume + cooldown) -> `retrained`
(déclenche le training DVC si la décision est positive).

Comme `training_pipeline.py`, ces assets appellent les MÊMES fonctions que
les CLI (`src/monitoring/`) : zéro duplication. Le sensor `drift_sensor`
évalue `evaluated` toutes les 30 minutes et déclenche le job dès que
`should_retrain` est vrai — le cooldown file-based évite les doublons.

Idempotent : rejouer le job régénère le rapport et réévalue la décision ;
le réentraînement n'a lieu que si l'arbitrage l'autorise.
"""

# NB : pas de `from __future__ import annotations` ici — Dagster valide les
# annotations de `context` au runtime (cf. pipelines/training_pipeline.py).

from collections.abc import Iterator

from dagster import (
    AssetExecutionContext,
    AssetSelection,
    Definitions,
    RunRequest,
    SensorEvaluationContext,
    SkipReason,
    asset,
    define_asset_job,
    sensor,
)
from src.config import load_params
from src.monitoring import drift_detector, retrain_policy


@asset(description="Rapport Evidently + résumé JSON sur la fenêtre d'inférences")
def drifted(context: AssetExecutionContext) -> dict:
    params = load_params()
    summary, html_path, json_path = drift_detector.run(params)
    context.log.info(
        f"dataset_drift={summary.dataset_drift} part={summary.drift_share:.2f} "
        f"n_cur={summary.n_current}"
    )
    return summary.to_dict() | {"html": str(html_path), "json": str(json_path)}


@asset(description="Arbitrage réentraînement (drift + volume + cooldown)")
def evaluated(context: AssetExecutionContext, drifted: dict) -> dict:
    decision = retrain_policy.decide(drifted)
    context.log.info(f"should_retrain={decision.should_retrain} : {'; '.join(decision.reasons)}")
    return decision.to_dict()


@asset(description="Réentraînement conditionnel (training DVC si arbitrage positif)")
def retrained(context: AssetExecutionContext, evaluated: dict) -> dict:
    if not evaluated.get("should_retrain"):
        return {"retrained": False, "reason": "; ".join(evaluated.get("reasons", []))}
    retrain_policy.record_retrain()
    from src.training import train

    metrics = train.run(load_params())
    context.log.info(f"réentraînement effectué : {metrics}")
    return {"retrained": True, "metrics": metrics}


monitoring_job = define_asset_job(
    name="monitoring_job",
    selection=AssetSelection.assets("drifted", "evaluated", "retrained"),
)


@sensor(job=monitoring_job, minimum_interval_seconds=1800)
def drift_sensor(
    context: SensorEvaluationContext,
) -> Iterator[RunRequest | SkipReason]:
    """Toutes les 30 min : si l'arbitrage demande un retrain, lance le job."""
    decision = retrain_policy.decide()
    if decision.should_retrain:
        yield RunRequest(tags={"trigger": "drift-sensor"})
    else:
        yield SkipReason(f"pas de réentraînement : {'; '.join(decision.reasons)}")


defs = Definitions(
    assets=[drifted, evaluated, retrained],
    jobs=[monitoring_job],
    sensors=[drift_sensor],
)
