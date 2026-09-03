"""Tests unitaires du générateur : déterminisme, qualité de base, modes drift."""

from __future__ import annotations

import pandas as pd
from src.data.generate_raw import generate


def test_generation_deterministe() -> None:
    pd.testing.assert_frame_equal(generate(200, seed=42), generate(200, seed=42))


def test_base_saine_sans_null_ni_hors_plage() -> None:
    df = generate(2000, seed=42)
    assert df["monthly_fee"].isna().sum() == 0
    assert (df["age"] >= 18).all() and (df["age"] <= 75).all()
    assert set(df["churn"].unique()) <= {0, 1}
    assert not df.isna().any().any()


def test_le_drift_shift_deplace_les_distributions() -> None:
    base = generate(2000, seed=42)
    drifted = generate(2000, seed=42, drift="shift")
    assert drifted["num_support_calls"].mean() > base["num_support_calls"].mean()
    assert drifted["tenure_months"].mean() < base["tenure_months"].mean()


def test_le_drift_corrupt_injecte_des_nulls_et_des_hors_plage() -> None:
    df = generate(2000, seed=42, drift="corrupt")
    assert df["monthly_fee"].isna().sum() > 0
    assert (df["age"] < 18).sum() > 0
    assert "unknown_value" in set(df["contract_type"].unique())
