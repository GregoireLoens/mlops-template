"""Pipeline Dagster du template : la vue « orchestration » du code src/.

DVC (dvc.yaml) reste le cache de données reproductible ; Dagster apporte
l'opérationnel (retries, scheduling, monitoring, lignée). Chaque asset
appelle LES MÊMES fonctions que dvc.yaml — zéro duplication, mêmes gates,
mêmes logs MLflow. L'évaluation fait partie de l'asset `trained` (métriques
test + log MLflow + registre).

Chaîne : prepare -> validate (GE, bloquante) -> materialize Feast
         -> train + evaluate -> promote (gates tests + métriques).

Idempotent : rejouer le job ré-exécute les gates et régénère des sorties
déterministes ; la promotion n'applique que si les gates passent.
"""

# NB : pas de `from __future__ import annotations` ici — Dagster valide les
# annotations de `context` au runtime et ne résout pas les strings (PEP 563).

from dagster import AssetExecutionContext, AssetSelection, Definitions, asset, define_asset_job
from src.config import load_params
from src.data import prepare, validate
from src.features.store import materialize_latest
from src.training import promote, train


@asset(description="Split train/test stratifié (étape DVC prepare)")
def prepared(context: AssetExecutionContext) -> dict:
    train_df, test_df = prepare.run(load_params())
    context.log.info(f"train={len(train_df)} test={len(test_df)}")
    return {"n_train": len(train_df), "n_test": len(test_df)}


@asset(description="Gate Great Expectations — lève DataValidationError si rouge")
def validated(prepared: dict) -> dict:
    validate.run(load_params())
    return prepared


@asset(description="Materialize du feature store Feast vers l'online store")
def featurized(validated: dict) -> dict:
    materialize_latest()
    return validated


@asset(description="Training + évaluation + log MLflow + alias challenger")
def trained(featurized: dict) -> dict:
    return train.run(load_params())


@asset(description="Promotion challenger->prod (gates : tests modèle + métriques)")
def promoted(trained: dict) -> dict:
    decision = promote.run(load_params())
    return {"promoted": decision.applied, "reason": decision.reason}


# Sélection par AssetSelection.assets (JAMAIS une string) : dagster parserait
# la string via ANTLR, dont le runtime est figé à 4.9 par dvc -> omegaconf.
training_job = define_asset_job(
    name="training_job",
    selection=AssetSelection.assets("prepared", "validated", "featurized", "trained", "promoted"),
)

defs = Definitions(
    assets=[prepared, validated, featurized, trained, promoted],
    jobs=[training_job],
)
