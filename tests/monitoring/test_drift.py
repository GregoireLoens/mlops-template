"""Tests de détection de drift : nominal => False, dérivé => True + colonnes.

Le détecteur compare une fenêtre courante à la baseline DVC
(`data/prepared/train.csv`). Le trafic nominal (même loi génératrice)
reste sous le seuil `share_threshold` ; le trafic data-drift (prix +25,
ancienneté raccourcie, support saturé) le franchit et identifie les
colonnes en faute. Fenêtres >= min_current_rows pour la significativité.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src.config import load_params
from src.data.generate_raw import generate
from src.monitoring import drift_detector as dd
from src.monitoring.simulate_production import build_frame, ground_truth


@pytest.fixture(scope="module")
def params():  # type: ignore[no-untyped-def]
    return load_params()


@pytest.fixture(scope="module")
def reference(params):  # type: ignore[no-untyped-def]
    return pd.read_csv(f"data/prepared/{'train.csv'}")


def _with_proba(df: pd.DataFrame, value: float) -> pd.DataFrame:
    out = df.copy()
    out["churn_probability"] = value
    return out


def test_nominal_pas_de_drift(reference: pd.DataFrame, params) -> None:  # type: ignore[no-untyped-def]
    cur = _with_proba(generate(2000, seed=123), 0.3)
    summary = dd.build_summary(reference, cur, params)
    assert summary.dataset_drift is False
    assert summary.drift_share <= params.monitoring.drift.share_threshold


def test_data_drift_alerte_et_colonnes(reference: pd.DataFrame, params) -> None:  # type: ignore[no-untyped-def]
    cur = _with_proba(build_frame("data-drift", 600, seed=42), 0.45)
    summary = dd.build_summary(reference, cur, params)
    assert summary.dataset_drift is True
    assert summary.drift_share > params.monitoring.drift.share_threshold
    for col in ("monthly_fee", "tenure_months", "num_support_calls"):
        assert col in summary.drifted_columns


def test_concept_drift_features_stables(  # type: ignore[no-untyped-def]
    reference: pd.DataFrame, params, tmp_path
) -> None:
    """Correctif 3 : une vérité dégradée fait passer `degraded` à True.

    Inférences nominales (features stables => pas de data drift) avec un
    prédicteur parfait pour la loi nominale, puis vérité retardée
    volontairement dégradée (labels inversés = concept drift extrême où la
    relation features->cible s'inverse). `evaluate_performance` doit signaler
    `degraded=True` avec une accuracy sous le seuil — le test échoue si la
    logique de détection est cassée (toujours False). Le contrôle nominal
    (non dégradé) garantit l'inverse (échoue si toujours True).
    Note : le générateur `concept-drift` réel ne suffit pas ici (80 %
    d'accord avec le nominal, AUC 0.77 > seuil) — l'inversion simule le
    changement de régime sévère que le détecteur doit impérativement voir.
    """
    from src.data.generate_raw import generate as _gen

    raw = _gen(600, seed=42)
    y_nominal = ground_truth(raw, "nominal", seed=42)
    # Prédicteur parfait pour le régime nominal (0.9 si churn, 0.1 sinon).
    cur = raw.copy()
    cur["churn_probability"] = np.where(y_nominal.to_numpy() == 1, 0.9, 0.1)
    cur["prediction_id"] = [f"test-{i:05d}" for i in range(len(cur))]
    summary = dd.build_summary(reference, cur, params)
    assert summary.dataset_drift is False
    # Contrôle : vérité nominale => pas dégradé.
    gt_nom_path = tmp_path / "gt_nominal.csv"
    with open(gt_nom_path, "w", encoding="utf-8") as f:
        f.write("prediction_id,churn_true\n")
        for pid, y in zip(cur["prediction_id"], y_nominal, strict=True):
            f.write(f"{pid},{int(y)}\n")
    res_nom = dd.evaluate_performance(cur, str(gt_nom_path), params)
    assert res_nom["evaluated"] is True
    assert res_nom["degraded"] is False
    # Vérité dégradée (concept drift) => dégradé, accuracy sous le seuil.
    gt_deg_path = tmp_path / "gt_degraded.csv"
    with open(gt_deg_path, "w", encoding="utf-8") as f:
        f.write("prediction_id,churn_true\n")
        for pid, y in zip(cur["prediction_id"], 1 - y_nominal.to_numpy(), strict=True):
            f.write(f"{pid},{int(y)}\n")
    res_deg = dd.evaluate_performance(cur, str(gt_deg_path), params)
    assert res_deg["evaluated"] is True
    assert res_deg["join"] == "prediction_id"
    assert res_deg["degraded"] is True
    assert res_deg["accuracy"] < params.monitoring.performance.min_accuracy
    assert res_deg["roc_auc"] < params.monitoring.performance.min_roc_auc


def test_fenetre_trop_petite_avertit(reference: pd.DataFrame, params) -> None:  # type: ignore[no-untyped-def]
    cur = _with_proba(generate(10, seed=1), 0.3)
    summary = dd.build_summary(reference, cur, params)
    assert any("trop petite" in w for w in summary.warnings)


def test_colonnes_manquantes_ignorees(reference: pd.DataFrame, params) -> None:  # type: ignore[no-untyped-def]
    cur = _with_proba(generate(600, seed=5), 0.3).drop(columns=["age"])
    summary = dd.build_summary(reference, cur, params)
    assert "age" not in [c["column"] for c in summary.columns]


def test_prediction_drift_calcule(reference: pd.DataFrame, params) -> None:  # type: ignore[no-untyped-def]
    """Correctif 4 : écart de moyenne au lieu du KS continu-vs-binaire.

    Comparer churn_probability (continue) à une cible binaire bruitée via KS
    n'est pas sain (masse en deux points vs distribution continue). Choix :
    écart absolu des moyennes, sans hypothèse distributionnelle (PSI écarté :
    pas de distribution de référence des probas + binning fragile). Le nominal
    (proba 0.30 vs taux ~0.28) reste sous le seuil, une dérive réelle (proba
    0.95) le franchit — échoue si la détection est cassée (toujours True/False).
    """
    cur_nominal = _with_proba(generate(600, seed=9), 0.30)
    summary_nominal = dd.build_summary(reference, cur_nominal, params)
    assert summary_nominal.prediction_drift is False
    assert 0.0 <= summary_nominal.prediction_pvalue <= 1.0

    cur_shifted = _with_proba(generate(600, seed=9), 0.95)
    summary_shifted = dd.build_summary(reference, cur_shifted, params)
    assert summary_shifted.prediction_drift is True
    assert summary_shifted.prediction_pvalue > 0.10
    assert 0.0 <= summary_shifted.prediction_pvalue <= 1.0
    # Traçabilité de la méthode (le champ pvalue porte le gap, cf. compute_drifts).
    methods = {c["column"]: c["method"] for c in summary_shifted.columns}
    assert methods.get("churn_probability") == "mean_gap"


def test_simulateur_modes_deterministes() -> None:
    a = build_frame("nominal", 100, seed=7)
    b = build_frame("nominal", 100, seed=7)
    pd.testing.assert_frame_equal(a, b)
    drifted = build_frame("data-drift", 100, seed=7)
    assert drifted["monthly_fee"].mean() > a["monthly_fee"].mean() + 10
    assert drifted["tenure_months"].mean() < a["tenure_months"].mean()
    with pytest.raises(ValueError, match="mode inconnu"):
        build_frame("nope", 10, seed=0)


def test_ground_truth_format() -> None:
    df = build_frame("nominal", 50, seed=3)
    gt = ground_truth(df, "nominal", seed=3)
    assert set(gt.unique()) <= {0, 1}
    assert len(gt) == 50
    assert gt.name == "churn_true"
    assert not np.isnan(gt.to_numpy(dtype=float)).any()


def test_evaluate_performance_jointure_prediction_id(params, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Correctif 2 (Option A) : la jointure sur prediction_id survit au mélange.

    Fenêtre courante ordonnée (10x proba 0.1 puis 10x 0.9) face à une vérité
    mélangée (ordre inverse) : l'ancien alignement positionnel donnerait
    accuracy 0.0, la jointure exacte donne 1.0. Échoue si evaluate_performance
    réaligne par position.
    """
    ids = [f"pid-{i:03d}" for i in range(20)]
    probas = [0.1] * 10 + [0.9] * 10
    cur = pd.DataFrame({"churn_probability": probas, "prediction_id": ids})
    gt_path = tmp_path / "gt.csv"
    with open(gt_path, "w", encoding="utf-8") as f:
        f.write("prediction_id,churn_true\n")
        for i in reversed(range(20)):
            f.write(f"{ids[i]},{0 if i < 10 else 1}\n")
    res = dd.evaluate_performance(cur, str(gt_path), params)
    assert res["evaluated"] is True
    assert res["join"] == "prediction_id"
    assert res["n"] == 20
    assert res["accuracy"] == 1.0


def test_write_ground_truth_ids_reels(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Correctif 2 (Option A) : le simulateur écrit les IDs réellement servis."""
    from src.monitoring.simulate_production import write_ground_truth

    df = build_frame("nominal", 5, seed=3)
    pids: list[str | None] = ["uuid-a", "uuid-b", None, "uuid-d", "uuid-e"]
    out = tmp_path / "gt.csv"
    write_ground_truth(str(out), df, "nominal", 3, pids)
    saved = pd.read_csv(out)
    assert list(saved.columns) == ["prediction_id", "churn_true"]
    # La requête en échec (None) est exclue : 4 lignes, IDs réels préservés.
    assert saved["prediction_id"].tolist() == ["uuid-a", "uuid-b", "uuid-d", "uuid-e"]
