"""Étape DVC `validate` : gate bloquante Great Expectations entre prepare et train.

Valide raw, train, test contre leurs suites ; échoue avec un message clair
(liste des expectations violées) si une donnée est douteuse. Le rapport
HTML (data docs) est publié dans reports/data_docs — un OUT DVC, donc
versionné via dvc.lock et reproductible par run.

Principe BP2 : une donnée douteuse n'atteint jamais le training. Le rapport
est construit AVANT la levée d'exception, pour rester consultable sur un
pipeline rouge.

Le store `gx/` est commité (config) ; suites/assets/definitions n'y sont
ré-écrits qu'en cas de changement sémantique réel (hors UUID) : un
`make repro` sans changement de code/params laisse `git status` clean.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import great_expectations as gx
import pandas as pd
from great_expectations.data_context import AbstractDataContext
from src.config import PROJECT_ROOT, Params, load_params
from src.data.expectations import (
    PREPARED_SUITE,
    RAW_SUITE,
    build_prepared_suite,
    build_raw_suite,
)
from src.data.prepare import read_raw

GX_DIR = PROJECT_ROOT / "gx"
REPORTS_DIR = PROJECT_ROOT / "reports"

DATASOURCE = "local_pandas"
BATCH_DEFINITION = "whole"

# label -> (asset, suite) ; un checkpoint par dataset, réutilisé à chaque run.
DATASETS = {
    "raw": (RAW_SUITE, "raw_data"),
    "train": (PREPARED_SUITE, "train_data"),
    "test": (PREPARED_SUITE, "test_data"),
}


class DataValidationError(RuntimeError):
    """Gate GE violée : bloque dvc repro et les jobs Dagster."""


def get_context() -> AbstractDataContext:
    """Contexte file persistant dans gx/ (config commitée, auto-réparée)."""
    conf = GX_DIR / "great_expectations.yml"
    if not conf.exists():
        # Bootstrap en sous-processus : get_context se met en cache PAR
        # PROCESSUS — un bootstrap in-process figerait le config template
        # avant notre patch (progress_bars), qui serait ignoré.
        bootstrap = (
            "import great_expectations as gx; "
            f"gx.get_context(mode='file', context_root_dir=r'{GX_DIR}')"
        )
        subprocess.run([sys.executable, "-c", bootstrap], check=True, capture_output=True)
    _ensure_progress_bars_off(conf)
    return gx.get_context(mode="file", context_root_dir=str(GX_DIR))


def _ensure_progress_bars_off(conf: Path) -> None:
    """Patch idempotent du yml : sortie de pipeline sobre (pas de barres tqdm)."""
    text = conf.read_text(encoding="utf-8")
    if "progress_bars" not in text:
        text += "\n# Template : sortie de pipeline sobre.\nprogress_bars:\n  globally: false\n"
        conf.write_text(text, encoding="utf-8")


def _publish_data_docs(ctx: AbstractDataContext) -> Path:
    """Publie le site data docs dans reports/data_docs — artefact du pipeline.

    Le site canonique de GX vit dans gx/uncommitted/ (détail interne, variable
    selon les versions). On en publie une copie stable dans reports/, un OUT
    DVC : le rapport HTML est versionné via dvc.lock et remonte en PR.
    """
    ctx.build_data_docs()
    dst = REPORTS_DIR / "data_docs"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(GX_DIR / "uncommitted" / "data_docs", dst)
    # Le nom du sous-dossier site (local_site/) est un détail GX : on résout
    # l'index réel plutôt que de coder le chemin en dur.
    return next(dst.glob("**/index.html"))


def _sans_ids(obj: Any) -> Any:
    """Contenu sémantique d'un objet GX sérialisé (sans les UUID volatils)."""
    if isinstance(obj, dict):
        return {k: _sans_ids(v) for k, v in obj.items() if k != "id"}
    if isinstance(obj, list):
        return [_sans_ids(v) for v in obj]
    return obj


def _build_suite_offline(build_fn: Any, params: Params) -> gx.ExpectationSuite:
    """Construit une suite SANS polluer le vrai store.

    Avec un contexte actif et un nom déjà stocké, chaque `add_expectation`
    persiste immédiatement dans `gx/` (doublons à chaque run). On construit
    donc dans un contexte jetable (jamais lu), puis on restaure le projet
    réel — le store commité n'est touché que sur changement réel (cf.
    `_ensure_suite`).
    """
    scratch_gx = Path(tempfile.gettempdir()) / "mlops-gx-build" / "gx"
    scratch_gx.mkdir(parents=True, exist_ok=True)
    gx.get_context(mode="file", context_root_dir=str(scratch_gx))
    try:
        return build_fn(params)
    finally:
        gx.get_context(mode="file", context_root_dir=str(GX_DIR))


def _ensure_suite(
    ctx: AbstractDataContext, name: str, built: gx.ExpectationSuite
) -> gx.ExpectationSuite:
    """Suite stockée si identique, sinon sauvegarde (changement réel, une fois)."""
    try:
        stored = ctx.suites.get(name)
    except Exception:
        stored = None
    if stored is not None and _sans_ids(stored.to_json_dict()) == _sans_ids(built.to_json_dict()):
        return stored
    return ctx.suites.add_or_update(built)


def _ensure_datasource(ctx: AbstractDataContext) -> tuple[Any, bool]:
    """Datasource + assets + batch definitions, créés seulement si absents.

    Ne JAMAIS `add_or_update` une datasource existante : GX y réécrit tout
    le mapping (nouveaux UUID, voire assets perdus) à chaque run.

    Retourne (datasource, changed) : après un heal (changed=True), les objets
    GX mis en cache par le contexte restent périmés — les définitions de
    validation sont alors réécrites une fois (cf. `force`), puis le régime
    nominal ne touche plus à rien.
    """
    changed = False
    try:
        ds: Any = ctx.data_sources.get(DATASOURCE)
    except Exception:
        ds = ctx.data_sources.add_pandas(DATASOURCE)
        changed = True
    for _label, (_suite_name, asset_name) in DATASETS.items():
        if asset_name in ds.get_asset_names():
            asset = ds.get_asset(asset_name)
        else:
            asset = ds.add_dataframe_asset(asset_name)
            changed = True
        try:
            asset.get_batch_definition(BATCH_DEFINITION)
        except (KeyError, gx.exceptions.DataContextError):
            asset.add_batch_definition_whole_dataframe(BATCH_DEFINITION)
            changed = True
    return ds, changed


def _ensure_validation_definition(
    ctx: AbstractDataContext,
    label: str,
    batch_def: Any,
    suite: gx.ExpectationSuite,
    force: bool = False,
) -> Any:
    """ValidationDefinition existante si câblage identique, sinon (re)création."""
    vd_name = f"{label}_vd"
    candidate = gx.ValidationDefinition(name=vd_name, data=batch_def, suite=suite)
    if not force:
        try:
            stored = ctx.validation_definitions.get(vd_name)
        except Exception:
            stored = None
        if stored is not None and _sans_ids(stored.dict()) == _sans_ids(candidate.dict()):
            return stored
    return ctx.validation_definitions.add_or_update(candidate)


def _checkpoint_for(
    ctx: AbstractDataContext,
    label: str,
    batch_def: Any,
    suite: gx.ExpectationSuite,
    force_vd_update: bool = False,
) -> gx.Checkpoint:
    """Get-or-create idempotent : validation definition -> checkpoint."""
    vd = _ensure_validation_definition(ctx, label, batch_def, suite, force=force_vd_update)
    return ctx.checkpoints.add_or_update(
        gx.Checkpoint(name=f"{label}_checkpoint", validation_definitions=[vd])
    )


def _run_validation(
    ctx: AbstractDataContext,
    label: str,
    df: pd.DataFrame,
    suite: gx.ExpectationSuite,
    batch_def: Any,
    force_vd_update: bool = False,
) -> gx.core.validation_definition.ValidationDefinitionResult:  # type: ignore[name-defined]
    cp = _checkpoint_for(ctx, label, batch_def, suite, force_vd_update=force_vd_update)
    cp_result = cp.run(batch_parameters={"dataframe": df})
    return list(cp_result.run_results.values())[0]


def _failure_message(label: str, result: Any) -> str:
    total = len(result.results)
    failed = [r for r in result.results if not r.success]
    lines = [f"[{label}] {len(failed)}/{total} expectations violées :"]
    for r in failed:
        cfg = r.expectation_config
        col = cfg.kwargs.get("column", "-")
        detail = ""
        res = r.result or {}
        if res.get("unexpected_count") is not None:
            detail = f" -> {res['unexpected_count']} valeurs inattendues"
        if res.get("observed_value") is not None:
            detail += f" (observé : {res['observed_value']})"
        lines.append(f"  - {cfg.type} [column={col}]{detail}")
    return "\n".join(lines)


def run(params: Params) -> None:
    """Valide les 3 datasets, publie le rapport HTML, lève si la gate est rouge."""
    ctx = get_context()
    suites = {
        RAW_SUITE: _ensure_suite(ctx, RAW_SUITE, _build_suite_offline(build_raw_suite, params)),
        PREPARED_SUITE: _ensure_suite(
            ctx, PREPARED_SUITE, _build_suite_offline(build_prepared_suite, params)
        ),
    }
    ds, ds_changed = _ensure_datasource(ctx)
    batch_defs = {
        label: ds.get_asset(asset_name).get_batch_definition(BATCH_DEFINITION)
        for label, (_suite_name, asset_name) in DATASETS.items()
    }

    frames = {
        "raw": read_raw(PROJECT_ROOT / params.data.raw_path),
        # parse_dates : même dtype que raw, sinon les expectations temporelles
        # comparent des chaînes à des Timestamps (piège classique de batch).
        "train": pd.read_csv(PROJECT_ROOT / params.data.train_path, parse_dates=["signup_ts"]),
        "test": pd.read_csv(PROJECT_ROOT / params.data.test_path, parse_dates=["signup_ts"]),
    }

    failures: list[str] = []
    for label, df in frames.items():
        suite_name, _ = DATASETS[label]
        result = _run_validation(
            ctx, label, df, suites[suite_name], batch_defs[label], force_vd_update=ds_changed
        )
        if result.success:
            print(f"validate OK — {label} ({len(df)} lignes)")
        else:
            failures.append(_failure_message(label, result))

    # Rapport HTML versionné (out DVC reports/) — généré même en échec.
    report = _publish_data_docs(ctx)
    print(f"rapport GE : {report}")

    if failures:
        raise DataValidationError("\n".join(failures))


def main() -> None:
    logging.getLogger("great_expectations").setLevel(logging.WARNING)
    try:
        run(load_params())
    except DataValidationError as exc:
        # Message clair pour le pipeline (dvc/Dagster/CI) : quoi, où, combien.
        print(f"\n[PIPELINE BLOQUÉ — gate GE]\n{exc}\n", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
