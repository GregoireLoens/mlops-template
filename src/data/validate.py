"""Étape DVC `validate` : gate bloquante Great Expectations entre prepare et train.

Valide raw, train, test contre leurs suites ; échoue avec un message clair
(liste des expectations violées) si une donnée est douteuse. Le rapport
HTML (data docs) est publié dans reports/data_docs — un OUT DVC, donc
versionné via dvc.lock et reproductible par run.

Principe BP2 : une donnée douteuse n'atteint jamais le training. Le rapport
est construit AVANT la levée d'exception, pour rester consultable sur un
pipeline rouge.

Le store `gx/` est commité (config) ; suites/assets/checkpoints y sont
ré-enregistrés de façon idempotente à chaque run (add_or_update).
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
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


def _checkpoint_for(
    ctx: AbstractDataContext, label: str, suite: gx.ExpectationSuite
) -> gx.Checkpoint:
    """Get-or-create idempotent : datasource -> asset -> batch def -> checkpoint."""
    ds = ctx.data_sources.add_or_update_pandas(name=DATASOURCE)
    _, asset_name = DATASETS[label]
    asset = (
        ds.get_asset(asset_name)
        if asset_name in ds.get_asset_names()
        else ds.add_dataframe_asset(asset_name)
    )
    try:
        batch_def = asset.get_batch_definition(BATCH_DEFINITION)
    except (KeyError, gx.exceptions.DataContextError):
        batch_def = asset.add_batch_definition_whole_dataframe(BATCH_DEFINITION)

    # add_or_update plutôt que get + mutation : les objets GX sont figés
    # (pydantic), le store reste cohérent avec la définition en code.
    vd = ctx.validation_definitions.add_or_update(
        gx.ValidationDefinition(name=f"{label}_vd", data=batch_def, suite=suite)
    )
    return ctx.checkpoints.add_or_update(
        gx.Checkpoint(name=f"{label}_checkpoint", validation_definitions=[vd])
    )


def _run_validation(
    ctx: AbstractDataContext, label: str, df: pd.DataFrame, suite: gx.ExpectationSuite
) -> gx.core.validation_definition.ValidationDefinitionResult:  # type: ignore[name-defined]
    cp = _checkpoint_for(ctx, label, suite)
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
        RAW_SUITE: ctx.suites.add_or_update(build_raw_suite(params)),
        PREPARED_SUITE: ctx.suites.add_or_update(build_prepared_suite(params)),
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
        result = _run_validation(ctx, label, df, suites[suite_name])
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
