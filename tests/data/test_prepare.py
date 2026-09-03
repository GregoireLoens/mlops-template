"""Tests unitaires du split : proportions, stratification, déterminisme."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from src.data.prepare import read_raw, split


def _fake_df(n: int = 1000) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "customer_id": [f"C{i:04d}" for i in range(n)],
            "age": rng.integers(18, 75, n),
            "churn": rng.binomial(1, 0.3, n),
            "signup_ts": pd.date_range("2025-01-01", periods=n, freq="h"),
        }
    )


def test_split_respecte_la_taille_de_test() -> None:
    train_df, test_df = split(_fake_df(1000), "churn", 0.2, 42)
    assert len(test_df) == 200
    assert len(train_df) == 800
    # Pas de fuite : aucune ligne ne passe dans les deux côtés.
    assert set(train_df["customer_id"]).isdisjoint(set(test_df["customer_id"]))


def test_split_est_stratifie_sur_la_cible() -> None:
    df = _fake_df(1000)
    train_df, test_df = split(df, "churn", 0.2, 42)
    base = df["churn"].mean()
    assert abs(test_df["churn"].mean() - base) < 0.05
    assert abs(train_df["churn"].mean() - base) < 0.05


def test_split_est_deterministe() -> None:
    df = _fake_df(1000)
    (a_train, a_test) = split(df, "churn", 0.2, 42)
    (b_train, b_test) = split(df, "churn", 0.2, 42)
    pd.testing.assert_frame_equal(a_train, b_train)
    pd.testing.assert_frame_equal(a_test, b_test)


def test_read_raw_parse_les_timestamps(tmp_path: Path) -> None:
    p = tmp_path / "raw.csv"
    p.write_text("customer_id,signup_ts,age,churn\nC00001,2025-03-01,42,0\n")
    df = read_raw(p)
    assert str(df["signup_ts"].dtype).startswith("datetime64")
