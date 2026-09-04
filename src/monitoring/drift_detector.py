"""Détection de drift données + prédiction avec Evidently (BP3, étape 3).

Compare la fenêtre d'inférences récentes (courant) à la baseline
d'entraînement DVC (référence, `data/prepared/train.csv`) :

- **Data drift** : Evidently `DataDriftPreset` (KS pour continues, Chi-2 /
  Z-test pour catégorielles, seuil p-value paramétrable). Le verdict global
  suit la règle `share_of_drifted_columns > share_threshold` (défaut 0.3).
- **Prediction drift** : écart absolu de moyenne |mean(churn_probability
  courant) - mean(cible entraînement)| > 0.10 — une dérive de la sortie sans
  dérive d'entrée est le signal d'un concept drift naissant (le KS
  continu-vs-binaire est volontairement écarté, cf. `compute_drifts`).
- **Performance différée** : si un CSV de vérité terrain (`prediction_id`,
  `churn_true`) est fourni, jointure réelle sur `prediction_id` (left join
  de la vérité sur les inférences) puis accuracy/F1/ROC-AUC courantes vs
  métriques du training (`metrics.json`) — dégradation > tolérance =>
  concept drift confirmé. Sans `prediction_id` commun (CSV legacy), repli
  positionnel documenté dans `evaluate_performance`.

Sorties (`reports/monitoring/`, paramétrable) :
- `drift_report.html` : rapport Evidently autonome (interactif) ;
- `drift_summary.json` : bilan synthétique (`dataset_drift`, colonnes en
  faute, parts, verdicts performance) — lu par les gauges Prometheus, la
  politique de réentraînement et le sensor Dagster.

Sans Evidently installé (extra `monitoring` absent), le détecteur bascule
sur un fallback scipy-free (KS/Chi-2 via numpy) : le rapport HTML est une
page minimale mais le JSON reste contractuel pour les tests et le retrain.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from src.config import PROJECT_ROOT, Params, load_params
from src.serving.inference_log import latest_inference_frames

PREDICTION_COL = "churn_probability"


@dataclass(frozen=True)
class ColumnDrift:
    column: str
    drifted: bool
    pvalue: float
    method: str


@dataclass
class DriftSummary:
    generated_at: str = ""
    n_reference: int = 0
    n_current: int = 0
    dataset_drift: bool = False
    drift_share: float = 0.0
    drifted_columns: list[str] = field(default_factory=list)
    columns: list[dict[str, Any]] = field(default_factory=list)
    prediction_drift: bool = False
    prediction_pvalue: float = 1.0
    target_drift: bool = False
    performance: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _feature_columns(params: Params) -> tuple[list[str], list[str]]:
    return list(params.features.numeric), list(params.features.categorical)


def load_reference(params: Params) -> pd.DataFrame:
    """Baseline d'entraînement (DVC) : features + cible."""
    path = PROJECT_ROOT / params.monitoring.reference_path
    return pd.read_csv(path)


def load_current(
    params: Params,
    current_path: str | None = None,
    limit: int = 10_000,
) -> pd.DataFrame:
    """Fenêtre d'inférences : CSV/JSONL explicite ou logs JSONL récents.

    Les lignes de log portent `features` (dict) + `churn_probability` ; on
    les aplati en colonnes. Un CSV passé via `--current` (ex. export du
    simulateur) est accepté tel quel s'il contient les features.
    """
    if current_path:
        path = Path(current_path)
        if path.suffix == ".jsonl":
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        else:
            return pd.read_csv(path)
    else:
        base = PROJECT_ROOT / params.monitoring.inferences_dir
        rows = latest_inference_frames(base, limit=limit)
    if not rows:
        return pd.DataFrame()
    flat = [
        {
            **r.get("features", {}),
            PREDICTION_COL: r.get("churn_probability"),
            # Préserve la clé de jointure pour evaluate_performance (Option A :
            # jointure réelle sur prediction_id, pas d'alignement positionnel).
            **({"prediction_id": r["prediction_id"]} if "prediction_id" in r else {}),
        }
        for r in rows
    ]
    return pd.DataFrame(flat)


def _ks_pvalue(ref: pd.Series, cur: pd.Series) -> float:
    """KS approximé sans scipy : statistique D -> p-value via Kolmogorov."""
    import numpy as np

    a = np.sort(ref.dropna().to_numpy(dtype=float))
    b = np.sort(cur.dropna().to_numpy(dtype=float))
    if len(a) == 0 or len(b) == 0:
        return 1.0
    grid = np.sort(np.concatenate([a, b]))
    cdf_a = np.searchsorted(a, grid, side="right") / len(a)
    cdf_b = np.searchsorted(b, grid, side="right") / len(b)
    d = float(np.max(np.abs(cdf_a - cdf_b)))
    n_eff = len(a) * len(b) / (len(a) + len(b))
    lam = (np.sqrt(n_eff) + 0.12 + 0.11 / np.sqrt(n_eff)) * d
    # Série de Kolmogorov (Marsaglia) : P(D > d) sous H0.
    pvalue = 0.0
    for k in range(1, 101):
        term = (-1) ** (k - 1) * np.exp(-2 * (k**2) * (lam**2))
        pvalue += term
        if abs(term) < 1e-12:
            break
    return float(min(1.0, max(0.0, 2 * pvalue)))


def _chi2_pvalue(ref: pd.Series, cur: pd.Series) -> float:
    """Chi-2 d'homogénéité sans scipy (approximation normale du quantile)."""
    from math import erf, sqrt

    import numpy as np

    cats = sorted(set(ref.dropna().unique()) | set(cur.dropna().unique()))
    if len(cats) <= 1:
        return 1.0
    table = np.array(
        [
            [(ref == c).sum() for c in cats],
            [(cur == c).sum() for c in cats],
        ],
        dtype=float,
    )
    row_sums = table.sum(axis=1, keepdims=True)
    col_sums = table.sum(axis=0, keepdims=True)
    total = table.sum()
    expected = row_sums @ col_sums / total
    mask = expected > 0
    chi2 = float((((table - expected) ** 2)[mask] / expected[mask]).sum())
    dof = (table.shape[0] - 1) * (table.shape[1] - 1)
    if dof <= 0:
        return 1.0
    # Wilson-Hilferty : chi2 -> normale standard, puis survie gaussienne.
    z = ((chi2 / dof) ** (1 / 3) - (1 - 2 / (9 * dof))) / sqrt(2 / (9 * dof))
    return float(1 - 0.5 * (1 + erf(z / sqrt(2))))


def _column_drift(
    column: str, ref: pd.Series, cur: pd.Series, pvalue_threshold: float, categorical: bool
) -> ColumnDrift:
    method = "chi2" if categorical else "ks"
    pvalue = _chi2_pvalue(ref, cur) if categorical else _ks_pvalue(ref, cur)
    return ColumnDrift(
        column=column, drifted=bool(pvalue < pvalue_threshold), pvalue=pvalue, method=method
    )


def compute_drifts(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    params: Params,
) -> tuple[list[ColumnDrift], ColumnDrift | None]:
    """Drift par colonne (features) + drift de prédiction, méthode documentée."""
    numeric, categorical = _feature_columns(params)
    threshold = params.monitoring.drift.pvalue_threshold
    columns: list[ColumnDrift] = []
    for col in numeric + categorical:
        if col not in reference.columns or col not in current.columns:
            continue
        columns.append(
            _column_drift(col, reference[col], current[col], threshold, col in categorical)
        )
    pred_drift: ColumnDrift | None = None
    if PREDICTION_COL in current.columns and params.data.target in reference.columns:
        # Correctif 4 revue BP3 : écart de moyenne au lieu du KS continu-vs-binaire.
        # L'ancien code comparait churn_probability (continue) à la cible binaire
        # bruitée N(0, 1e-3) via KS — comparer une distribution continue à une
        # masse en deux points n'est pas un test sain (p-value artificiellement
        # basse, sensible au jitter, non interprétable). Choix : écart absolu
        # |mean(proba courante) - mean(cible référence)| — sans hypothèse
        # distributionnelle, robuste sur petites fenêtres, interprétable
        # (un modèle calibré a un écart ~0 en nominal). PSI écarté : il exige
        # une distribution de référence des probas (absente — seule la cible
        # binaire existe) + binning fragile. Seuil 0.10 : le nominal (0.30 vs
        # ~0.28 => gap ~0.02) reste vert, une dérive réelle (0.95 vs ~0.28 =>
        # gap ~0.67) lève l'alerte. Le champ `pvalue` porte ici le gap [0, 1]
        # (compat JSON, toujours dans [0, 1]) avec `method="mean_gap"`.
        ref_mean = float(reference[params.data.target].mean())
        cur_mean = float(current[PREDICTION_COL].mean())
        gap = abs(cur_mean - ref_mean)
        pred_drift = ColumnDrift(
            column=PREDICTION_COL,
            drifted=bool(gap > 0.10),
            pvalue=float(gap),
            method="mean_gap",
        )
    return columns, pred_drift


def evaluate_performance(
    current: pd.DataFrame,
    ground_truth_path: str | None,
    params: Params,
) -> dict[str, Any]:
    """Métriques différées courantes vs training (concept drift confirmé ?).

    Choix revue BP3 — Option A (recommandée) : jointure réelle sur
    `prediction_id` (left join de la vérité terrain sur les inférences).
    Le simulateur écrit les `prediction_id` réellement servis par l'API et
    `load_current` préserve cette colonne depuis les logs JSONL, donc
    l'appariement survit aux réordonnancements, aux requêtes perdues et aux
    fenêtres partielles — contrairement à l'ancien alignement positionnel
    `n = min(len(gt), len(current))`. Repli positionnel conservé uniquement
    quand l'une des deux tables n'a pas de `prediction_id` (CSV legacy avec
    `row_id`, ou frame synthétique des tests) ; le champ `join` expose le
    mode utilisé (`prediction_id` vs `positional`) pour la traçabilité.
    """
    if not ground_truth_path:
        return {"evaluated": False, "reason": "pas de vérité terrain fournie"}
    gt = pd.read_csv(ground_truth_path)
    if "churn_true" not in gt.columns:
        return {"evaluated": False, "reason": "colonne churn_true absente"}
    if PREDICTION_COL not in current.columns:
        return {"evaluated": False, "reason": "pas de prédictions dans la fenêtre"}
    # Normalise l'ID legacy `row_id` vers `prediction_id` quand c'est la
    # seule clé disponible côté vérité terrain.
    if "prediction_id" not in gt.columns and "row_id" in gt.columns:
        gt = gt.rename(columns={"row_id": "prediction_id"})
    join_mode = "positional"
    if "prediction_id" in gt.columns and "prediction_id" in current.columns:
        merged = current.merge(gt[["prediction_id", "churn_true"]], on="prediction_id", how="inner")
        if len(merged) == 0:
            return {
                "evaluated": False,
                "reason": "aucune correspondance prediction_id entre inférences et vérité terrain",
            }
        y_true = merged["churn_true"].to_numpy()
        proba = merged[PREDICTION_COL].to_numpy(dtype=float)
        n = len(merged)
        join_mode = "prediction_id"
    else:
        n = min(len(gt), len(current))
        y_true = gt["churn_true"].to_numpy()[:n]
        proba = current[PREDICTION_COL].to_numpy(dtype=float)[:n]
    pred = (proba >= 0.5).astype(int)
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    metrics: dict[str, Any] = {
        "evaluated": True,
        "join": join_mode,
        "n": n,
        "accuracy": float(accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)) if len(set(y_true)) > 1 else 0.5,
    }
    baseline = _training_baseline()
    metrics["baseline"] = baseline
    drop = baseline.get("roc_auc", 1.0) - metrics["roc_auc"]
    metrics["roc_auc_drop"] = float(drop)
    degraded = (
        metrics["roc_auc"] < params.monitoring.performance.min_roc_auc
        or drop > params.monitoring.performance.drop_tolerance
    )
    metrics["degraded"] = bool(degraded)
    return metrics


def _training_baseline() -> dict[str, float]:
    try:
        metrics = json.loads((PROJECT_ROOT / "metrics.json").read_text(encoding="utf-8"))
        return {k: float(metrics[k]) for k in ("accuracy", "f1", "roc_auc") if k in metrics}
    except (OSError, json.JSONDecodeError, ValueError, TypeError):
        return {}


def build_summary(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    params: Params,
    ground_truth_path: str | None = None,
) -> DriftSummary:
    """Assemble le verdict : parts, colonnes, prédiction, performance."""
    summary = DriftSummary(
        generated_at=datetime.now(UTC).isoformat(),
        n_reference=len(reference),
        n_current=len(current),
    )
    min_cur = params.monitoring.drift.min_current_rows
    min_ref = params.monitoring.drift.min_reference_rows
    if len(current) < min_cur:
        summary.warnings.append(
            f"fenêtre courante trop petite ({len(current)} < {min_cur}) : verdict indicatif"
        )
    if len(reference) < min_ref:
        summary.warnings.append(
            f"référence trop petite ({len(reference)} < {min_ref}) : verdict indicatif"
        )
    columns, pred_drift = compute_drifts(reference, current, params)
    summary.columns = [asdict(c) for c in columns]
    drifted = [c.column for c in columns if c.drifted]
    summary.drifted_columns = drifted
    summary.drift_share = len(drifted) / len(columns) if columns else 0.0
    summary.dataset_drift = bool(summary.drift_share > params.monitoring.drift.share_threshold)
    if pred_drift is not None:
        summary.prediction_drift = pred_drift.drifted
        summary.prediction_pvalue = pred_drift.pvalue
        summary.columns.append(asdict(pred_drift))
    summary.performance = evaluate_performance(current, ground_truth_path, params)
    return summary


def _evidently_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    params: Params,
) -> str | None:
    """Rapport HTML Evidently (None si l'extra `monitoring` est absent)."""
    try:
        from evidently import Dataset, Report
        from evidently.core.datasets import DataDefinition
        from evidently.presets import DataDriftPreset
    except ImportError:
        return None
    numeric, categorical = _feature_columns(params)
    cols = [c for c in numeric + categorical if c in current.columns]
    ref = reference[[c for c in numeric + categorical if c in reference.columns]].copy()
    cur = current[cols].copy()
    # La cible binaire n'est pas une feature : on l'exclut du preset drift.
    # `churn_probability` (courant seul) est aussi exclue : Evidently exige
    # des colonnes présentes des deux côtés — la dérive de prédiction est
    # calculée séparément (KS, cf. compute_drifts) et exposée dans le JSON.
    report = Report([DataDriftPreset()])
    snapshot = report.run(
        Dataset.from_pandas(cur, data_definition=DataDefinition()),
        Dataset.from_pandas(ref, data_definition=DataDefinition()),
    )
    return snapshot.get_html_str(as_iframe=False)


def write_outputs(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    summary: DriftSummary,
    params: Params,
) -> tuple[Path, Path]:
    """Écrit `drift_report.html` + `drift_summary.json` (retourne les chemins)."""
    out_dir = PROJECT_ROOT / params.monitoring.reports_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / "drift_report.html"
    json_path = out_dir / "drift_summary.json"
    html = _evidently_report(reference, current, params)
    if html is None:
        html = _fallback_html(summary)
    html_path.write_text(html, encoding="utf-8")
    json_path.write_text(json.dumps(summary.to_dict(), indent=2) + "\n", encoding="utf-8")
    return html_path, json_path


def _fallback_html(summary: DriftSummary) -> str:
    rows = "\n".join(
        f"<tr><td>{c['column']}</td><td>{c['method']}</td>"
        f"<td>{c['pvalue']:.4f}</td><td>{'DRIFT' if c['drifted'] else 'ok'}</td></tr>"
        for c in summary.columns
    )
    return f"""<!DOCTYPE html><html lang="fr"><head><meta charset="utf-8">
<title>Rapport de drift (fallback, Evidently absent)</title></head><body>
<h1>Drift — {summary.generated_at}</h1>
<p>dataset_drift={summary.dataset_drift} (part={summary.drift_share:.2f}), """
    f"""n_ref={summary.n_reference}, n_cur={summary.n_current}</p>
<table border="1"><tr><th>colonne</th><th>méthode</th><th>p-value</th><th>verdict</th></tr>
{rows}</table>
<p>Installez l'extra <code>monitoring</code> pour le rapport Evidently interactif.</p>
</body></html>"""


def run(
    params: Params,
    current_path: str | None = None,
    ground_truth_path: str | None = None,
    limit: int = 10_000,
) -> tuple[DriftSummary, Path, Path]:
    """Pipeline complet : charge, compare, évalue, écrit les rapports."""
    reference = load_reference(params)
    current = load_current(params, current_path=current_path, limit=limit)
    if current.empty:
        raise RuntimeError(
            "fenêtre d'inférences vide : appelez /predict ou passez --current CSV/JSONL"
        )
    summary = build_summary(reference, current, params, ground_truth_path)
    html_path, json_path = write_outputs(reference, current, summary, params)
    print(
        f"[drift] dataset_drift={summary.dataset_drift} "
        f"(part={summary.drift_share:.2f}, n_cur={summary.n_current}) "
        f"-> {html_path}, {json_path}"
    )
    return summary, html_path, json_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", default=None, help="CSV/JSONL de la fenêtre courante")
    parser.add_argument("--ground-truth", default=None, help="CSV (prediction_id, churn_true)")
    parser.add_argument("--limit", type=int, default=10_000)
    args = parser.parse_args()
    run(load_params(), current_path=args.current, ground_truth_path=args.ground_truth)


if __name__ == "__main__":
    main()
