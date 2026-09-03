"""Tests de comportement directionnel : bouger une feature clé doit déplacer
la prédiction dans le bon sens (le domaine métier est encodé dans le test).

Garantis seulement pour la régression logistique (coefficients signés) :
pour un RF, passer à des tests d'importance/explicabilité (SHAP) — hors
scope template. Skip automatique si model_type != logreg.
"""

from __future__ import annotations

import pandas as pd
import pytest


def _proba_for(model, base: dict, **overrides) -> float:
    row = pd.DataFrame([{**base, **overrides}])
    return float(model.predict_proba(row)[0, 1])


@pytest.fixture()
def base_row(test_df, features) -> dict:
    return test_df[features].iloc[0].to_dict()


def _skip_si_pas_logreg(params) -> None:
    if params.train.model_type != "logreg":
        pytest.skip("tests directionnels : garantis pour logreg uniquement")


def test_plus_de_support_calls_augmente_le_risque(model, base_row, params) -> None:
    _skip_si_pas_logreg(params)
    plus_de_calls = int(base_row["num_support_calls"]) + 2
    assert _proba_for(model, base_row, num_support_calls=plus_de_calls) > _proba_for(
        model, base_row
    )


def test_tenure_plus_courte_augmente_le_risque(model, base_row, params) -> None:
    _skip_si_pas_logreg(params)
    tenure_courte = max(0.0, float(base_row["tenure_months"]) - 6)
    assert _proba_for(model, base_row, tenure_months=tenure_courte) > _proba_for(model, base_row)


def test_month_to_month_plus_risque_que_contrat_deux_ans(model, base_row, params) -> None:
    _skip_si_pas_logreg(params)
    m2m = _proba_for(model, base_row, contract_type="month_to_month")
    two_years = _proba_for(model, base_row, contract_type="two_year")
    assert m2m > two_years


def test_premium_reduit_le_risque(model, base_row, params) -> None:
    _skip_si_pas_logreg(params)
    if int(base_row["has_premium"]) == 1:
        base_row["has_premium"] = 0
    assert _proba_for(model, base_row, has_premium=1) < _proba_for(model, base_row)
