# src/data — génération, préparation, validation

- `generate_raw.py` : dataset de churn simulé (aucune donnée réelle). Modes `--drift shift|corrupt` pour déformer volontairement les distributions — utilisés pour tester la gate GE et le monitoring (BP3).
- `prepare.py` : étape DVC `prepare` — split train/test stratifié déterministe. Le préprocessing vit dans le Pipeline sklearn de l'étape `train` (zéro skew train/serving).
- `expectations.py` : suites GE **en code** (raw + prepared) — la source de vérité, revue en PR ; le store `gx/` n'est qu'un cache idempotent (add_or_update).
- `validate.py` : étape DVC `validate` — gate bloquante (message clair par colonne), rapport HTML diagnostique dans `reports/data_docs` (régénéré, non versionné), sentinelle `reports/validate.ok` (out DVC, dép de `train`).
- `../config.py` : `params.yaml` chargé en objets typés, partagé par DVC, Dagster et les tests.
