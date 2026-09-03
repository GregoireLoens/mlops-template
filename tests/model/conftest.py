"""Fixtures partagées des tests modèle : modèle challengé + jeu de test.

Session-scoped : un seul chargement pour tous les tests du dossier.
Skip propre si le modèle n'existe pas encore (repo fraîchement cloné
sans `make repro`) — les tests modèle supposent un training passé.
"""

from __future__ import annotations

import pandas as pd
import pytest
from src.config import PROJECT_ROOT, Params, load_params
from src.training.registry import load_challenger
from src.training.train import MODEL_PATH


@pytest.fixture(scope="session")
def params() -> Params:
    return load_params()


@pytest.fixture(scope="session")
def model_source(params: Params) -> str:
    if not MODEL_PATH.exists():
        pytest.skip("models/model.pkl absent — lancer `make train` d'abord")
    _, source = load_challenger(params)
    return source


@pytest.fixture(scope="session")
def model(model_source: str, params: Params):
    model, _ = load_challenger(params)
    return model


@pytest.fixture(scope="session")
def test_df(params: Params) -> pd.DataFrame:
    return pd.read_csv(PROJECT_ROOT / params.data.test_path)


@pytest.fixture(scope="session")
def features(params: Params) -> list[str]:
    return params.features.numeric + params.features.categorical
