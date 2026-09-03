"""Génération d'un dataset de churn simulé — aucune donnée réelle externe.

Le dataset est déterministe (seed) et imite un contexte SaaS B2C :
- numériques   : age, tenure_months, monthly_fee, num_support_calls, has_premium
- catégorielles: contract_type, signup_channel
- cible        : churn, pilotée par une logistique sur les features
  (signal fort : num_support_calls, tenure, contract month-to-month).

Les modes --drift déforment volontairement les données :
- `shift`   : déplacement de distribution (support calls, tenure, fees) —
  sert à tester la détection de drift (GE étape 3, monitoring BP3) ;
- `corrupt` : nulls + valeurs hors plage + catégorie inconnue — doit faire
  échouer la gate de validation avec un message clair.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from src.config import PROJECT_ROOT, load_params

COLUMNS = [
    "customer_id",
    "signup_ts",
    "age",
    "tenure_months",
    "monthly_fee",
    "num_support_calls",
    "has_premium",
    "contract_type",
    "signup_channel",
    "churn",
]


def generate(n_rows: int, seed: int, drift: str = "none") -> pd.DataFrame:
    if drift not in {"none", "shift", "corrupt"}:
        raise ValueError(f"drift inconnu : {drift!r} (attendu : none|shift|corrupt)")

    rng = np.random.default_rng(seed)
    n = n_rows

    contract_type = rng.choice(["month_to_month", "one_year", "two_year"], n, p=[0.6, 0.25, 0.15])
    has_premium = rng.binomial(1, 0.3, n)
    num_support_calls = rng.poisson(1.5, n)
    tenure_months = np.clip(rng.exponential(24, n), 0, 120)
    monthly_fee = np.clip(rng.normal(29, 10, n), 5, 120)

    if drift == "shift":
        # Dérive réaliste : support saturé, clients plus récents, hausse des prix.
        num_support_calls = rng.poisson(4.0, n)
        tenure_months = tenure_months * 0.6
        monthly_fee = monthly_fee + 15.0

    churn_logit = (
        -1.6
        + 0.55 * num_support_calls
        - 0.04 * tenure_months
        + 0.9 * (contract_type == "month_to_month")
        + 0.02 * (monthly_fee - 29)
        - 0.3 * has_premium
        + rng.normal(0, 0.5, n)
    )
    churn = rng.binomial(1, 1 / (1 + np.exp(-churn_logit)))

    df = pd.DataFrame(
        {
            "customer_id": [f"C{i:05d}" for i in range(n)],
            "signup_ts": pd.Timestamp("2025-01-01")
            + pd.to_timedelta(rng.integers(0, 180, n), unit="D"),
            "age": np.clip(rng.normal(40, 12, n), 18, 75).round(),
            "tenure_months": tenure_months.round(1),
            "monthly_fee": monthly_fee.round(2),
            "num_support_calls": num_support_calls,
            "has_premium": has_premium,
            "contract_type": contract_type,
            "signup_channel": rng.choice(["web", "mobile", "referral"], n, p=[0.5, 0.35, 0.15]),
            "churn": churn,
        }
    )[COLUMNS]

    if drift == "corrupt":
        # 1% de nulls, valeurs hors plage, catégorie inconnue :
        # la suite GE doit bloquer le pipeline sur ces anomalies.
        fee_na = rng.choice(n, size=max(1, n // 100), replace=False)
        df.loc[fee_na, "monthly_fee"] = np.nan
        bad_age = rng.choice(n, size=max(1, n // 500), replace=False)
        df.loc[bad_age, "age"] = -1
        bad_cat = rng.choice(n, size=max(1, n // 500), replace=False)
        df.loc[bad_cat, "contract_type"] = "unknown_value"

    return df


def main() -> None:
    params = load_params()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=params.data.seed)
    parser.add_argument("--drift", choices=["none", "shift", "corrupt"], default="none")
    parser.add_argument(
        "--out", type=str, default=params.data.raw_path, help="Chemin relatif au repo"
    )
    args = parser.parse_args()

    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = generate(args.rows, args.seed, args.drift)
    df.to_csv(out_path, index=False)
    print(f"généré : {out_path} ({len(df)} lignes, drift={args.drift})")


if __name__ == "__main__":
    main()
