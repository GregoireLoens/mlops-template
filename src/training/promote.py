"""Promotion challenger -> champion (alias `prod`) — gate + journal d'audit.

Règle de promotion (toutes conditions requises) :
1. **tests modèle verts** (`tests/model`, exécutés ici : seuils `eval` +
   invariances + comportement directionnel) ;
2. challenger STRICTEMENT meilleur que le champion sur la métrique de
   référence (roc_auc) — pas d'amélioration mesurable, pas de promotion.

La décision est journalisée dans reports/promotions.md (qui / quoi / quand /
métriques) : fichier commité, c'est l'audit trail du template.
Première promotion : pas de champion -> le challenger devient prod.

`rollback` repointe `prod` vers la version précédente (lue dans le journal).
"""

from __future__ import annotations

import getpass
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime

import mlflow
from mlflow.tracking import MlflowClient
from src.config import PROJECT_ROOT, Params, load_params

PROMOTIONS_LOG = PROJECT_ROOT / "reports" / "promotions.md"
REF_METRIC = "roc_auc"  # métrique de comparaison challenger/champion

_HEADER = """# Journal des promotions

Une ligne par décision : qui, quoi, quand, quelles métriques.
`prod` = alias champion (servi en production), `challenger` = dernier entraîné.

> Le journal est un audit trail local : le registre MLflow est la source de
> vérité des alias. En cas de doute (journal désynchronisé d'un registre
> reconstruit), vérifier avec `MlflowClient().get_model_version_by_alias`.

| Date (UTC) | Qui | Modèle | Challenger | Champion | Décision | Motif |
| --- | --- | --- | --- | --- | --- | --- |
"""


@dataclass(frozen=True)
class Decision:
    promoted: bool
    challenger_version: str
    champion_version: str | None
    challenger_metric: float
    champion_metric: float | None
    tests_ok: bool
    reason: str
    applied: bool = False


def _setup() -> None:
    # Serveur down : fail fast (sinon backoff exponentiel MLflow, minutes).
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "2")
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if uri:
        mlflow.set_tracking_uri(uri)


def _metric_of(client: MlflowClient, version: str, name: str) -> float:
    mv = client.get_model_version_by_alias(name, version)
    assert mv is not None and mv.run_id is not None, f"version invalide : {name}@{version}"
    run = client.get_run(mv.run_id)
    return float(run.data.metrics[REF_METRIC])


def _run_model_tests() -> tuple[bool, str]:
    """Gate : les tests modèle chargent le challenger (registry) et doivent passer."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/model", "-q", "--no-header", "-x"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    tail = (result.stdout + result.stderr).strip().splitlines()
    return result.returncode == 0, tail[-1] if tail else ""


def decide(params: Params) -> Decision:
    _setup()
    client = MlflowClient()
    model_name = params.train.model_name

    challenger = client.get_model_version_by_alias(model_name, "challenger")
    challenger_metric = _metric_of(client, "challenger", model_name)

    try:
        champion_metric = _metric_of(client, "prod", model_name)
        champion_version = "prod"
    except Exception:
        champion_metric, champion_version = None, None

    tests_ok, tests_detail = _run_model_tests()
    if not tests_ok:
        return Decision(
            promoted=False,
            challenger_version=challenger.version,
            champion_version=champion_version,
            challenger_metric=challenger_metric,
            champion_metric=champion_metric,
            tests_ok=False,
            reason=f"tests modèle en échec ({tests_detail})",
        )

    if champion_metric is None:
        return Decision(
            promoted=True,
            challenger_version=challenger.version,
            champion_version=None,
            challenger_metric=challenger_metric,
            champion_metric=None,
            tests_ok=True,
            reason="première promotion (pas de champion)",
        )

    if challenger_metric > champion_metric:
        return Decision(
            promoted=True,
            challenger_version=challenger.version,
            champion_version=champion_version,
            challenger_metric=challenger_metric,
            champion_metric=champion_metric,
            tests_ok=True,
            reason=f"{REF_METRIC} challenger > champion",
        )

    return Decision(
        promoted=False,
        challenger_version=challenger.version,
        champion_version=champion_version,
        challenger_metric=challenger_metric,
        champion_metric=champion_metric,
        tests_ok=True,
        reason=f"pas d'amélioration mesurable ({REF_METRIC} challenger <= champion)",
    )


def run(params: Params, dry_run: bool = False) -> Decision:
    decision = decide(params)
    applied = decision.promoted and not dry_run
    if applied:
        _setup()
        MlflowClient().set_registered_model_alias(
            name=params.train.model_name, alias="prod", version=decision.challenger_version
        )
    _journal(params, decision, applied)
    return Decision(**{**decision.__dict__, "applied": applied})


def _journal(params: Params, d: Decision, applied: bool) -> None:
    PROMOTIONS_LOG.parent.mkdir(parents=True, exist_ok=True)
    if not PROMOTIONS_LOG.exists():
        PROMOTIONS_LOG.write_text(_HEADER, encoding="utf-8")
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M")
    who = os.getenv("PROMOTED_BY", getpass.getuser())
    verdict = "PROMU" if applied else ("SIMULÉ" if d.promoted else "REFUSÉ")
    # Contexte registre : le journal se relit seul — on fige la run source du
    # challenger et l'alias prod AVANT décision (état réel, pas reconstruit).
    client = MlflowClient()
    challenger_mv = client.get_model_version_by_alias(params.train.model_name, "challenger")
    assert challenger_mv is not None and challenger_mv.run_id is not None
    challenger_run = client.get_run(challenger_mv.run_id).data.tags.get("mlflow.runName", "?")
    try:
        prod_before = client.get_model_version_by_alias(params.train.model_name, "prod").version
    except Exception:
        prod_before = "—"
    line = (
        f"| {now} | {who} | {params.train.model_name} "
        f"| v{d.challenger_version} ({challenger_run}, {REF_METRIC}={d.challenger_metric:.4f}) "
        f"| {d.champion_version or '—'} (prod=v{prod_before}, {REF_METRIC}="
        f"{f'{d.champion_metric:.4f}' if d.champion_metric is not None else '—'}) "
        f"| {verdict} | {d.reason} |\n"
    )
    with open(PROMOTIONS_LOG, "a", encoding="utf-8") as f:
        f.write(line)
    print(
        f"[promotion] {verdict} — challenger v{d.challenger_version} "
        f"({REF_METRIC}={d.challenger_metric:.4f}) vs champion "
        f"{d.champion_version or '—'} ({REF_METRIC}={d.champion_metric}) : {d.reason}"
    )


def rollback(params: Params) -> str:
    """Repointe `prod` vers la version précédente (lue dans le journal).

    Sécurité : la version cible est VÉRIFIÉE dans le registre avant le
    repointage — un registre reconstruit (vide) lève une erreur claire au
    lieu de repointer un alias vers une version fantôme.
    """
    _setup()
    client = MlflowClient()
    try:
        current = client.get_model_version_by_alias(params.train.model_name, "prod").version
    except Exception as exc:
        raise RuntimeError(
            "aucun alias prod dans le registre (serveur reconstruit ?) : "
            "relancer `make train` puis `make promote` pour recréer la lignée"
        ) from exc
    if not PROMOTIONS_LOG.exists():
        raise RuntimeError("journal des promotions absent : rien à rollbacker")
    entries = [
        line
        for line in PROMOTIONS_LOG.read_text(encoding="utf-8").splitlines()
        if line.startswith("|") and "PROMU" in line
    ]
    if len(entries) < 1:
        raise RuntimeError("aucune promotion dans le journal : rien à rollbacker")
    # Dernière version promue = cible du rollback de l'avant-dernière (ou v1).
    versions = [line.split("|")[4].strip().split(" ")[0] for line in entries]
    previous = next((v for v in reversed(versions) if v != f"v{current}"), None)
    if previous is None:
        raise RuntimeError(f"pas de version antérieure à v{current} dans le journal")
    previous = previous.lstrip("v")
    # Vérification registre : la version cible doit exister, sinon l'alias
    # pointerait une version absente et le serving refuserait de démarrer.
    try:
        client.get_model_version(params.train.model_name, previous)
    except Exception as exc:
        raise RuntimeError(
            f"version v{previous} absente du registre (serveur reconstruit ?) : "
            "rollback impossible, relancer `make train` puis `make promote`"
        ) from exc
    client.set_registered_model_alias(name=params.train.model_name, alias="prod", version=previous)
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M")
    with open(PROMOTIONS_LOG, "a", encoding="utf-8") as f:
        f.write(
            f"| {now} | {os.getenv('PROMOTED_BY', getpass.getuser())} "
            f"| {params.train.model_name} | — | v{previous} | ROLLBACK "
            f"| prod repointé de v{current} vers v{previous} |\n"
        )
    print(f"[rollback] prod : v{current} -> v{previous}")
    return previous


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="décision sans application")
    parser.add_argument("--rollback", action="store_true", help="repointe prod en arrière")
    args = parser.parse_args()
    params = load_params()
    if args.rollback:
        rollback(params)
    else:
        run(params, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
