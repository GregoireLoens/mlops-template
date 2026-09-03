"""Tests de la gate GE : passe sur données saines, bloque les corrompues.

Les tests utilisent un contexte GX isolé (tmp_path) : les stores du repo
(gx/) ne sont jamais pollués par les exécutions de test.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from src.config import load_params
from src.data import validate
from src.data.generate_raw import generate
from src.data.validate import DataValidationError


@pytest.fixture()
def ctx_isole(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirige le store GX et les rapports vers tmp_path pour le test."""
    monkeypatch.setattr(validate, "GX_DIR", tmp_path / "gx")
    monkeypatch.setattr(validate, "REPORTS_DIR", tmp_path / "reports")


def test_la_gate_passe_sur_les_donnees_du_repo(ctx_isole: None) -> None:
    # run() valide raw + train + test réels et publie le rapport ; ne lève pas.
    validate.run(load_params())


def test_la_gate_bloque_avec_un_message_clair(ctx_isole: None, tmp_path: Path) -> None:
    corrupt_csv = tmp_path / "raw_corrupt.csv"
    generate(1000, seed=3, drift="corrupt").to_csv(corrupt_csv, index=False)

    params = load_params()
    params = replace(params, data=replace(params.data, raw_path=str(corrupt_csv)))

    with pytest.raises(DataValidationError) as err:
        validate.run(params)

    msg = str(err.value)
    # Le message nomme le dataset, la colonne fautive et compte les violations.
    assert "[raw]" in msg
    assert "monthly_fee" in msg
    assert "expectations violées" in msg


def test_le_drift_shift_est_bloque_par_la_distribution_churn(
    ctx_isole: None, tmp_path: Path
) -> None:
    shifted_csv = tmp_path / "raw_shift.csv"
    generate(1000, seed=3, drift="shift").to_csv(shifted_csv, index=False)

    params = load_params()
    params = replace(params, data=replace(params.data, raw_path=str(shifted_csv)))

    with pytest.raises(DataValidationError) as err:
        validate.run(params)
    assert "churn" in str(err.value)
