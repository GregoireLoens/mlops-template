"""Tests d'invariance : des perturbations SANS sémantique ne doivent pas
changer la prédiction.

On ne teste que des invariances garanties par construction (ordre du batch,
duplication, cast numérique) — pas des transformations qui changeraient
réellement l'entrée du modèle (ex. renommer une catégorie).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def test_l_ordre_des_lignes_ne_change_pas_les_predictions(model, test_df, features) -> None:
    X = test_df[features].head(500)
    proba = model.predict_proba(X)[:, 1]
    shuffled = model.predict_proba(X.iloc[::-1])[:, 1]
    assert (shuffled == proba[::-1]).all()


def test_dupliquer_une_ligne_ne_change_pas_sa_prediction(model, test_df, features) -> None:
    X = test_df[features].head(100)
    duplicated = pd.concat([X, X.head(1)], ignore_index=True)
    original = model.predict_proba(X.head(1))[:, 1]
    again = model.predict_proba(duplicated.tail(1))[:, 1]
    assert (original == again).all()


def test_prediction_identique_en_solo_ou_en_batch(model, test_df, features) -> None:
    # Le serving (étape 10) prédit ligne par ligne : pas d'effet de contexte
    # batch. Tolérance flottante (BLAS matvec vs matmul) : < 1e-9.
    X = test_df[features].head(100)
    solo = model.predict_proba(X.head(1))[:, 1]
    in_batch = model.predict_proba(X)[:, 1][:1]
    assert np.allclose(solo, in_batch, atol=1e-9)


def test_cast_numerique_ne_change_pas_la_prediction(model, test_df, features) -> None:
    # 42 vs 42.0 : même valeur métier, même prédiction attendue.
    X_int = test_df[features].head(100)
    X_float = X_int.astype({c: "float64" for c in X_int.select_dtypes("int64").columns})
    assert (model.predict_proba(X_float)[:, 1] == model.predict_proba(X_int)[:, 1]).all()
