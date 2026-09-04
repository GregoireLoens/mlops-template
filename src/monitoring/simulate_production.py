"""Générateur de trafic de production + vérité terrain retardée (BP3, étape 2).

Bombarde le serving (nginx `:8090` ou FastAPI direct) selon 3 modes :

- `nominal` : lignes fidèles à la distribution d'entraînement (seed dédié) ;
- `data-drift` : même fonction cible, mais features déplacées (monthly_fee
  +25 €, tenure raccourcie, support saturé) — le drift de données doit lever
  l'alerte sans changer la relation features -> label ;
- `concept-drift` : features stables, mais la propension au churn change
  (logit inversé sur `has_premium` et `monthly_fee`) — seule la vérité
  retardée révèle la dégradation (concept drift pur).

Vérité retardée : `--with-ground-truth PATH` écrit un CSV
(prediction_id, churn_true) rejouant la vraie fonction génératrice — simule
les labels observés 30 jours plus tard pour l'évaluation différée.
Choix revue BP3 (Option A) : le simulateur réutilise les `prediction_id`
réellement servis par l'API (`/predict` les renvoie et les loggue), de sorte
que `evaluate_performance` peut faire une jointure exacte sur
`prediction_id` au lieu d'un alignement positionnel fragile. En `--dry-run`
(sans serveur), des IDs stables `sim-{mode}-{seed}-{i}` sont générés pour
garder la reproductibilité.

Le simulateur réutilise `src.data.generate_raw.generate` (même loi que le
training) : le mode nominal hérite du déterminisme (seed) du dataset.

Usage :
    python -m src.monitoring.simulate_production --mode nominal --n 500
    python -m src.monitoring.simulate_production --mode data-drift --n 500 \\
        --with-ground-truth /tmp/gt.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import urllib.error
import urllib.request
from dataclasses import dataclass

import numpy as np
import pandas as pd
from src.data.generate_raw import generate

Mode = str

# Échantillon envoyé au serveur : sous-ensemble des colonnes du dataset.
FEATURE_COLS = [
    "age",
    "tenure_months",
    "monthly_fee",
    "num_support_calls",
    "has_premium",
    "contract_type",
    "signup_channel",
]


@dataclass(frozen=True)
class TrafficResult:
    n_sent: int
    n_ok: int
    n_errors: int
    mean_proba: float


def build_frame(mode: Mode, n: int, seed: int) -> pd.DataFrame:
    """Construit les features à envoyer selon le mode (déterministe)."""
    if mode == "nominal":
        df = generate(n, seed, drift="none")
    elif mode == "data-drift":
        # Déplacement de distribution calqué sur le générateur : prix +25,
        # ancienneté raccourcie, support saturé — bornes Pydantic respectées.
        df = generate(n, seed, drift="none")
        df["monthly_fee"] = np.clip(df["monthly_fee"] + 25.0, 5, 200)
        df["tenure_months"] = np.clip(df["tenure_months"] * 0.4, 0, 120)
        df["num_support_calls"] = np.clip(df["num_support_calls"] + 3, 0, 50)
    elif mode == "concept-drift":
        df = generate(n, seed, drift="none")
    else:
        raise ValueError(f"mode inconnu : {mode!r} (nominal|data-drift|concept-drift)")
    return df


def ground_truth(df: pd.DataFrame, mode: Mode, seed: int) -> pd.Series:
    """Vrais labels rejoués hors-ligne (fonction génératrice connue).

    - nominal / data-drift : même logit que `generate_raw` ;
    - concept-drift : logit modifié (effet premium inversé, prix atténué,
      biais relevé) — les features ne bougent pas, la cible si.
    """
    rng = np.random.default_rng(seed + 999)
    is_mtm = (df["contract_type"] == "month_to_month").to_numpy(dtype=float)
    if mode == "concept-drift":
        logit = (
            -0.9
            - 0.5 * df["has_premium"].to_numpy(dtype=float)
            + 0.45 * df["num_support_calls"].to_numpy(dtype=float)
            - 0.02 * df["tenure_months"].to_numpy(dtype=float)
            + 0.4 * is_mtm
            - 0.01 * (df["monthly_fee"].to_numpy(dtype=float) - 29)
            + rng.normal(0, 0.5, len(df))
        )
    else:
        logit = (
            -1.6
            + 0.55 * df["num_support_calls"].to_numpy(dtype=float)
            - 0.04 * df["tenure_months"].to_numpy(dtype=float)
            + 0.9 * is_mtm
            + 0.02 * (df["monthly_fee"].to_numpy(dtype=float) - 29)
            - 0.3 * df["has_premium"].to_numpy(dtype=float)
            + rng.normal(0, 0.5, len(df))
        )
    proba = 1 / (1 + np.exp(-logit))
    return pd.Series(rng.binomial(1, proba), index=df.index, name="churn_true")


def _post_predict(url: str, payload: dict) -> tuple[bool, float, str | None]:
    req = urllib.request.Request(
        url.rstrip("/") + "/predict",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read().decode())
        # L'API renvoie le prediction_id servi (et loggué côté serving) :
        # on le propage pour que la vérité retardée soit joignable
        # exactement (Option A revue BP3), pas par position.
        pid = body.get("prediction_id")
        return True, float(body["churn_probability"]), str(pid) if pid else None
    except (urllib.error.HTTPError, urllib.error.URLError):
        return False, 0.0, None


def _row_payload(row: pd.Series) -> dict:
    base = {k: row[k] for k in FEATURE_COLS}
    base["age"] = int(base["age"])
    base["tenure_months"] = float(base["tenure_months"])
    base["monthly_fee"] = float(base["monthly_fee"])
    base["num_support_calls"] = int(base["num_support_calls"])
    base["has_premium"] = int(base["has_premium"])
    base["contract_type"] = str(base["contract_type"])
    base["signup_channel"] = str(base["signup_channel"])
    return base


def run_traffic(
    url: str, mode: Mode, n: int, seed: int
) -> tuple[TrafficResult, pd.DataFrame, list[str | None]]:
    """Envoie N requêtes et retourne (résultat, features, prediction_ids).

    Les `prediction_ids` sont les IDs réellement servis par l'API (None si
    la requête a échoué) — à passer à `write_ground_truth` pour une jointure
    exacte côté `evaluate_performance` (Option A revue BP3).
    """
    df = build_frame(mode, n, seed)
    ok, errors, probas = 0, 0, []
    prediction_ids: list[str | None] = []
    for _, row in df.iterrows():
        success, proba, pid = _post_predict(url, _row_payload(row))
        prediction_ids.append(pid)
        if success:
            ok += 1
            probas.append(proba)
        else:
            errors += 1
    result = TrafficResult(
        n_sent=n,
        n_ok=ok,
        n_errors=errors,
        mean_proba=float(np.mean(probas)) if probas else 0.0,
    )
    return result, df, prediction_ids


def write_ground_truth(
    path: str,
    df: pd.DataFrame,
    mode: Mode,
    seed: int,
    prediction_ids: list[str | None] | None = None,
) -> str:
    """Écrit le CSV de vérité retardée (prediction_id, churn_true).

    Choix revue BP3 (Option A) : quand les `prediction_ids` réellement servis
    par l'API sont fournis (via `run_traffic`), on les réutilise tels quels
    pour permettre la jointure exacte dans `evaluate_performance`. Sans IDs
    (`--dry-run` ou appel direct), repli sur des IDs stables déterministes
    `sim-{mode}-{seed}-{i}` (reproductibles, mais non joignables aux logs).
    Les requêtes en échec (prediction_id None) sont exclues : sans inférence
    logguée, leur label orphelin fausserait la jointure.
    """
    labels = ground_truth(df, mode, seed)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["prediction_id", "churn_true"])
        for i, label in enumerate(labels):
            if prediction_ids is not None:
                pid = prediction_ids[i] if i < len(prediction_ids) else None
                if pid is None:
                    continue
                writer.writerow([pid, int(label)])
            else:
                writer.writerow([f"sim-{mode}-{seed}-{i:05d}", int(label)])
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["nominal", "data-drift", "concept-drift"], required=True)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--url", default="http://localhost:8090")
    parser.add_argument("--with-ground-truth", default=None, help="chemin CSV de sortie")
    parser.add_argument(
        "--dry-run", action="store_true", help="génère les features sans appeler le serveur"
    )
    args = parser.parse_args()

    if args.dry_run:
        df = build_frame(args.mode, args.n, args.seed)
        print(f"[simulate] dry-run {args.mode} : {len(df)} lignes générées (aucun appel)")
        prediction_ids: list[str | None] | None = None
    else:
        result, df, prediction_ids = run_traffic(args.url, args.mode, args.n, args.seed)
        print(
            f"[simulate] {args.mode} via {args.url} : "
            f"{result.n_ok}/{result.n_sent} OK, {result.n_errors} erreurs, "
            f"proba moyenne={result.mean_proba:.3f}"
        )
    if args.with_ground_truth:
        out = write_ground_truth(args.with_ground_truth, df, args.mode, args.seed, prediction_ids)
        print(f"[simulate] vérité retardée : {out}")


if __name__ == "__main__":
    main()
