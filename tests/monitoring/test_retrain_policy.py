"""Tests de la politique de réentraînement : l'ordre n'est émis que si le
seuil est franchi (drift + volume + cooldown)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from src.config import load_params
from src.monitoring import retrain_policy


def _summary(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "dataset_drift": False,
        "drift_share": 0.0,
        "drifted_columns": [],
        "n_current": 0,
        "performance": {"evaluated": False},
    }
    base.update(overrides)
    return base


def test_pas_de_resume_pas_de_retrain() -> None:
    decision = retrain_policy.decide({})
    assert decision.should_retrain is False
    assert any("drift-report" in r for r in decision.reasons)


def test_nominal_pas_de_retrain() -> None:
    decision = retrain_policy.decide(_summary(n_current=5000))
    assert decision.should_retrain is False


def test_drift_sans_volume_pas_de_retrain() -> None:
    params = load_params()
    decision = retrain_policy.decide(
        _summary(
            dataset_drift=True,
            drift_share=0.5,
            drifted_columns=["monthly_fee"],
            n_current=params.monitoring.retrain.min_new_rows - 1,
        )
    )
    assert decision.should_retrain is False
    assert any("volume insuffisant" in r for r in decision.reasons)


def test_drift_plus_volume_declenche(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    params = load_params()
    monkeypatch.setattr(retrain_policy, "STATE_PATH", tmp_path / "state.json")
    decision = retrain_policy.decide(
        _summary(
            dataset_drift=True,
            drift_share=0.5,
            drifted_columns=["monthly_fee", "tenure_months"],
            n_current=params.monitoring.retrain.min_new_rows + 100,
        )
    )
    assert decision.should_retrain is True
    assert any("volume OK" in r for r in decision.reasons)


def test_cooldown_bloque_le_doublon(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    params = load_params()
    monkeypatch.setattr(retrain_policy, "STATE_PATH", tmp_path / "state.json")
    retrain_policy.record_retrain()  # arme le cooldown à l'instant
    decision = retrain_policy.decide(
        _summary(
            dataset_drift=True,
            drift_share=0.6,
            drifted_columns=["monthly_fee"],
            n_current=params.monitoring.retrain.min_new_rows + 1000,
        )
    )
    assert decision.should_retrain is False
    assert any("cooldown" in r for r in decision.reasons)


def test_degradation_sans_drift_declenche(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Concept drift pur : pas de drift données, mais perf dégradée + volume."""
    params = load_params()
    monkeypatch.setattr(retrain_policy, "STATE_PATH", tmp_path / "state.json")
    decision = retrain_policy.decide(
        _summary(
            n_current=params.monitoring.retrain.min_new_rows + 10,
            performance={"evaluated": True, "roc_auc": 0.5, "roc_auc_drop": 0.3, "degraded": True},
        )
    )
    assert decision.should_retrain is True
    assert any("dégradée" in r for r in decision.reasons)


def test_load_summary_absent() -> None:
    assert retrain_policy.load_summary(Path("/inexistant/summary.json")) == {}
