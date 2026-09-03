"""Étape DVC `prepare` : split train/test stratifié et déterministe.

Aucun encodage ni scaling ici : le préprocessing vit dans le Pipeline
sklearn de l'étape `train`, pour que l'objet packagé soit autonome — le
serving (étape 10) reçoit des features brutes et applique exactement les
mêmes transformations que le training (zéro skew train/serving).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import PROJECT_ROOT, Params, load_params


def read_raw(path: Path | str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["signup_ts"])
    return df


def split(
    df: pd.DataFrame, target: str, test_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df[target], random_state=seed
    )
    # Index réinitialisés : les deux fichiers restent stables d'un run à l'autre.
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def run(params: Params) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = read_raw(PROJECT_ROOT / params.data.raw_path)
    train_df, test_df = split(raw, params.data.target, params.data.test_size, params.data.seed)

    train_path = PROJECT_ROOT / params.data.train_path
    test_path = PROJECT_ROOT / params.data.test_path
    train_path.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    # Parquets pour la source Feast (feature view customer_profile) : même
    # contenu, format colonne avec dtypes préservés.
    train_df.to_parquet(train_path.with_suffix(".parquet"), index=False)
    test_df.to_parquet(test_path.with_suffix(".parquet"), index=False)
    return train_df, test_df


def main() -> None:
    params = load_params()
    train_df, test_df = run(params)
    print(
        f"prepare OK — train={len(train_df)} test={len(test_df)} "
        f"-> {params.data.train_path}, {params.data.test_path}"
    )


if __name__ == "__main__":
    main()
