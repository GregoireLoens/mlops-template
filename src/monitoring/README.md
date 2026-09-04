# src/monitoring — monitoring production, drift & retrain (BP3)

- `inference_log.py` : voir `src/serving/` — le logging vit côté serving.
- `simulate_production.py` : générateur de trafic (`nominal`, `data-drift`,
  `concept-drift`) + vérité terrain retardée (`--with-ground-truth`).
  `make simulate-traffic ARGS="--mode data-drift --n 500"`.
- `drift_detector.py` : Evidently `DataDriftPreset` (rapport HTML) + tests
  KS/Chi-2 maison (verdict JSON : `dataset_drift`, `drift_share`,
  `drifted_columns`, `prediction_drift`, `performance`). Seuils dans
  `params.yaml/monitoring`. `make drift-report ARGS="--current X.csv"`.
- `retrain_policy.py` : arbitrage drift + volume + cooldown.
  `make drift-check` (décision seule, exit 2 si retrain requis),
  `make retrain-if-drifted` (déclenche le training DVC — cron/CI).

Fenêtre d'inférences : logs JSONL (`data/inferences/`, via `/predict`) ou
CSV explicite (`--current`). Vérité retardée : CSV (`prediction_id`,
`churn_true`) joint par `prediction_id` aux inférences (Option A revue BP3 ;
repli positionnel uniquement pour les CSV legacy sans IDs).
