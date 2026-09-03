# data/ — versionnement DVC

- `raw/` : dataset de churn **simulé** (généré par `src/data/generate_raw.py`, déterministe). Versionné par `dvc add data/raw` → pointeur `raw.dvc` commité, contenu ignoré par git.
- `prepared/` : sorties de l'étape `prepare` (train/test CSV) — outs du pipeline `dvc.yaml`, tracées dans `dvc.lock`.
- Cache DVC : `.dvc/cache` (local). Remote distant (S3, MinIO…) : cf. `.env.example` (`DVC_REMOTE_URL`) et `docs/decisions.md`.

**Règle** : git suit les pointeurs (`*.dvc`, `dvc.lock`), DVC suit le contenu des données.
