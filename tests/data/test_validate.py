"""Tests de la gate GE : passe sur données saines, bloque les corrompues.

Hermétiques : les datasets sont générés dans tmp_path (jamais les CSV du
repo) — les tests passent sur clone frais sans `make repro`. Le store GX et
les rapports sont eux aussi isolés (tmp_path) : `gx/` n'est jamais pollué.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from src.config import Params, load_params
from src.data import validate
from src.data.generate_raw import generate
from src.data.prepare import split
from src.data.validate import DataValidationError

# La suite prepared exige >= 1000 lignes par split
# (ExpectTableRowCountToBeBetween) : 6000 brutes -> 4800 train / 1200 test.
N_RAW_SAIN = 6000


@pytest.fixture()
def ctx_isole(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirige le store GX et les rapports vers tmp_path pour le test."""
    monkeypatch.setattr(validate, "GX_DIR", tmp_path / "gx")
    monkeypatch.setattr(validate, "REPORTS_DIR", tmp_path / "reports")


@pytest.fixture()
def params_sains(tmp_path: Path) -> Params:
    """Jeu raw/train/test sain et cohérent (même loi que le pipeline DVC)."""
    params = load_params()
    raw = generate(N_RAW_SAIN, seed=42)
    train_df, test_df = split(raw, params.data.target, params.data.test_size, params.data.seed)
    raw_csv = tmp_path / "raw.csv"
    train_csv = tmp_path / "train.csv"
    test_csv = tmp_path / "test.csv"
    raw.to_csv(raw_csv, index=False)
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    return replace(
        params,
        data=replace(
            params.data,
            raw_path=str(raw_csv),
            train_path=str(train_csv),
            test_path=str(test_csv),
        ),
    )


def test_la_gate_passe_sur_donnees_saines(ctx_isole: None, params_sains: Params) -> None:
    # run() valide raw + train + test et publie le rapport ; ne lève pas.
    validate.run(params_sains)


def test_la_gate_bloque_avec_un_message_clair(
    ctx_isole: None, params_sains: Params, tmp_path: Path
) -> None:
    corrupt_csv = tmp_path / "raw_corrupt.csv"
    generate(1000, seed=3, drift="corrupt").to_csv(corrupt_csv, index=False)

    params = replace(params_sains, data=replace(params_sains.data, raw_path=str(corrupt_csv)))

    with pytest.raises(DataValidationError) as err:
        validate.run(params)

    msg = str(err.value)
    # Le message nomme le dataset, la colonne fautive et compte les violations.
    assert "[raw]" in msg
    assert "monthly_fee" in msg
    assert "expectations violées" in msg


def test_le_drift_shift_est_bloque_par_la_distribution_churn(
    ctx_isole: None, params_sains: Params, tmp_path: Path
) -> None:
    shifted_csv = tmp_path / "raw_shift.csv"
    generate(1000, seed=3, drift="shift").to_csv(shifted_csv, index=False)

    params = replace(params_sains, data=replace(params_sains.data, raw_path=str(shifted_csv)))

    with pytest.raises(DataValidationError) as err:
        validate.run(params)
    assert "churn" in str(err.value)
