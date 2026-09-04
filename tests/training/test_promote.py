"""Garde-fous de la promotion : erreurs claires, pas de tracebacks brutes."""

from __future__ import annotations

import pytest
from src.config import load_params
from src.training import promote


class _RegistreVide:
    """MlflowClient factice : aucun alias (serveur reconstruit / training skippé)."""

    def get_model_version_by_alias(self, *args: object, **kwargs: object) -> object:
        raise Exception("Registered model alias challenger not found")


def test_decide_sans_challenger_erreur_claire(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(promote, "MlflowClient", _RegistreVide)
    with pytest.raises(RuntimeError, match="aucun alias challenger"):
        promote.decide(load_params())
