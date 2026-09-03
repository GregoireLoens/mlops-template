# src/training — entraînement et packaging

- `train.py` : étape DVC `train` — Pipeline sklearn complet (préprocessing + estimateur), fit, évaluation sur test (accuracy / F1 / ROC-AUC), écrit `models/model.pkl` + `metrics.json` (métriques DVC, commité pour `dvc metrics diff`).
- Bascule d'estimateur par config seule : `train.model_type: logreg|rf` dans `params.yaml`.
- MLflow (params/metrics/artefacts/registre) sera branché sur ces mêmes fonctions à l'étape 5 : DVC, Dagster et la CI appellent le même code.
