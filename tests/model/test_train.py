"""Tests unitaires du builder de pipeline sklearn (préprocessing inclus)."""

from __future__ import annotations

from dataclasses import replace

import pytest
from src.config import load_params
from src.data.generate_raw import generate
from src.training.train import build_pipeline, train_model


def test_le_pipeline_contient_le_preprocessing_et_le_modele() -> None:
    model = build_pipeline(load_params())
    assert [name for name, _ in model.steps] == ["preprocess", "model"]


def test_model_type_inconnu_rejete() -> None:
    params = load_params()
    bad = replace(params, train=replace(params.train, model_type="xgboost"))
    with pytest.raises(ValueError, match="model_type inconnu"):
        build_pipeline(bad)


def test_fit_sur_donnees_simulees_et_probabilites_valides() -> None:
    params = load_params()
    df = generate(400, seed=42)
    model = train_model(df, params)
    proba = model.predict_proba(df[params.features.numeric + params.features.categorical])[:, 1]
    assert ((proba >= 0.0) & (proba <= 1.0)).all()
