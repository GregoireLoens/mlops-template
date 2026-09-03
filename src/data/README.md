# src/data — génération, préparation

- `generate_raw.py` : dataset de churn simulé (aucune donnée réelle). Modes `--drift shift|corrupt` pour déformer volontairement les distributions — utilisés pour tester la gate GE (étape 3) et le monitoring (BP3).
- `prepare.py` : étape DVC `prepare` — split train/test stratifié déterministe. Le préprocessing vit dans le Pipeline sklearn de l'étape `train` (zéro skew train/serving).
- `../config.py` : `params.yaml` chargé en objets typés, partagé par DVC, Dagster et les tests.
