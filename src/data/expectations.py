"""Suites d'expectations GE — définies en CODE, pas en JSON.

Principe : la source de vérité est ce fichier (revu en PR comme du code,
testé unitairement). À chaque run, les suites sont reconstruites puis
ré-enregistrées via `add_or_update` : le store `gx/` n'est qu'un cache.

- raw      : schéma exact, plages de valeurs, distribution clé (taux de churn) ;
- prepared : s'applique aux splits train/test — zéro null sur les features,
  plages conservées, stratification de la cible préservée.
"""

from __future__ import annotations

import great_expectations as gx
import pandas as pd
from src.config import Params

RAW_SUITE = "raw_core"
PREPARED_SUITE = "prepared_core"

# Bornes temporelles du dataset simulé (générateur : signup_ts sur 180 jours).
SIGNUP_MIN = pd.Timestamp("2025-01-01")
SIGNUP_MAX = pd.Timestamp("2025-06-30")

# Taux de churn simulé ~25% : bornes volontairement larges (0.10-0.40).
# Les dérives fines relèvent du monitoring (BP3), pas de la gate de batch.
CHURN_RATE_MIN = 0.10
CHURN_RATE_MAX = 0.40


def _schema_columns(params: Params) -> list[str]:
    return [
        "customer_id",
        "signup_ts",
        *params.features.numeric,
        *params.features.categorical,
        params.data.target,
    ]


def _add_domain_expectations(suite: gx.ExpectationSuite, params: Params) -> None:
    """Règles de domaine partagées raw / prepared : plages + ensembles + cible."""
    e = gx.expectations
    suite.add_expectation(
        e.ExpectColumnValuesToBeBetween(
            column="signup_ts", min_value=SIGNUP_MIN, max_value=SIGNUP_MAX
        )
    )
    suite.add_expectation(e.ExpectColumnValuesToBeBetween(column="age", min_value=18, max_value=75))
    suite.add_expectation(
        e.ExpectColumnValuesToBeBetween(column="tenure_months", min_value=0, max_value=120)
    )
    suite.add_expectation(
        e.ExpectColumnValuesToBeBetween(column="monthly_fee", min_value=5, max_value=120)
    )
    suite.add_expectation(
        e.ExpectColumnValuesToBeBetween(column="num_support_calls", min_value=0, max_value=20)
    )
    suite.add_expectation(e.ExpectColumnValuesToBeInSet(column="has_premium", value_set=[0, 1]))
    suite.add_expectation(
        e.ExpectColumnValuesToBeInSet(
            column="contract_type", value_set=["month_to_month", "one_year", "two_year"]
        )
    )
    suite.add_expectation(
        e.ExpectColumnValuesToBeInSet(
            column="signup_channel", value_set=["web", "mobile", "referral"]
        )
    )
    suite.add_expectation(e.ExpectColumnValuesToBeInSet(column="churn", value_set=[0, 1]))
    suite.add_expectation(
        e.ExpectColumnMeanToBeBetween(
            column="churn", min_value=CHURN_RATE_MIN, max_value=CHURN_RATE_MAX
        )
    )


def build_raw_suite(params: Params) -> gx.ExpectationSuite:
    """Suite des données brutes : schéma exact + règles de domaine + nulls maîtrisés."""
    suite = gx.ExpectationSuite(name=RAW_SUITE)
    e = gx.expectations

    # Le schéma est un contrat : une colonne ajoutée/renommée sans PR casse ici.
    suite.add_expectation(
        e.ExpectTableColumnsToMatchOrderedList(column_list=_schema_columns(params))
    )
    suite.add_expectation(e.ExpectColumnValuesToNotBeNull(column="customer_id"))
    suite.add_expectation(e.ExpectColumnValuesToBeUnique(column="customer_id"))
    suite.add_expectation(e.ExpectColumnValuesToNotBeNull(column="monthly_fee"))

    _add_domain_expectations(suite, params)
    return suite


def build_prepared_suite(params: Params) -> gx.ExpectationSuite:
    """Suite des splits train/test : zéro null partout, volumes minimaux."""
    suite = gx.ExpectationSuite(name=PREPARED_SUITE)
    e = gx.expectations

    suite.add_expectation(e.ExpectTableRowCountToBeBetween(min_value=1000))
    suite.add_expectation(e.ExpectColumnValuesToNotBeNull(column="customer_id"))
    suite.add_expectation(e.ExpectColumnValuesToBeUnique(column="customer_id"))
    # Aucune feature ne tolère un null une fois préparée : le training et le
    # serving doivent produire des prédictions déterministes sans imputation cachée.
    for col in [*_schema_columns(params)[2:], "customer_id"]:
        if col != params.data.target:
            suite.add_expectation(e.ExpectColumnValuesToNotBeNull(column=col))

    _add_domain_expectations(suite, params)
    return suite
