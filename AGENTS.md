# AGENTS.md — mlops-template

Guide pour tout agent (ou humain) qui modifie ce repo. Le `README.md` reste la
référence produit ; ce fichier impose les règles de travail.

## 1. C'est quoi ce projet

Template MLOps churn, agnostique cloud, construit étape par étape (l'historique
git est le support de formation) :

- **BP1** — training reproductible : DVC (données), Feast (features), MLflow
  (expériences/registre), Dagster (orchestration).
- **BP2** — CI/CD modèle : gate Great Expectations bloquante, tests modèle,
  promotion par alias MLflow, canary + rollback.
- **BP3** — monitoring & retrain loop : logs d'inférence JSONL, drift Evidently,
  performance différée, Prometheus, sensor Dagster → réentraînement auto.

Stack : Python **3.11** pin (`.python-version`), `uv` + `hatchling`, layout `src`
(`src.data`, `src.training`, … importés depuis la racine, `pythonpath = ["."]`).

## 2. Environnement et commandes — tout passe par `make`

```bash
make setup     # .venv (uv sync --extra dev) + .env + hooks pre-commit
make up        # stack locale : MLflow :5000 (docker compose, cf. docker/)
make data      # dataset simulé déterministe + `dvc add data/raw`
make train     # up + dvc repro : prepare -> gate GE -> train -> registre (challenger)
make repro     # `dvc repro` seul (sans relancer la stack)
make test      # pytest unitaires UNIQUEMENT (addopts `-m "not integration"`)
make test-integration  # bout-en-bout, lent
make lint      # ruff check + ruff format --check + mypy
make format    # ruff format + ruff check --fix
make precommit # pre-commit sur tout le repo
make promote / make rollback / make serve / make smoke / make serve-down
make drift-report / make drift-check / make retrain-if-drifted
make simulate-traffic ARGS="--mode data-drift --n 500"
make dagster-dev / make dagster-run / make monitoring-run / make monitoring-up
make health / make logs / make metrics / make help
```

Notes :

- Le `Makefile` exporte `.venv/bin` en tête de `PATH` : `python` dans `dvc.yaml`
  résout la venv sans activation manuelle. En shell manuel, utiliser
  `.venv/bin/python`, `.venv/bin/pytest`, etc.
- Ports : MLflow **:5000**, stable **:8001**, canary **:8002**, nginx **:8090**,
  Prometheus **:9090**.
- Sans `MLFLOW_TRACKING_URI`, MLflow écrit dans `./mlruns` (store fichier) : le
  pipeline reste reproductible sans serveur (c'est ce qu'utilise la CI).

## 3. Règles d'or (reproductibilité)

1. **Definition of Done** : `make up && make train` doit tourner **de zéro sans
   étape implicite**. Tout ce qui est régénérable (bases locales, caches,
   artefacts) est ignoré par git — git ne contient que code, config et
   pointeurs DVC.
2. **`params.yaml` est la source de vérité unique** (chemins, split, features,
   hyperparamètres, seuils). Il est en dépendance DVC → repro sélectif. Tout
   nouveau paramètre passe par `src/config.py` (dataclasses typées, échec tôt
   si champ manquant), jamais en dur.
3. **Ne jamais committer** : `.venv/`, `.env`, contenu de `data/*`, `mlruns/`,
   `/models`, **`mlflow.db`**, **`.tmp_dagster_home*/`**, `/reports/data_docs`,
   `/data/inferences`, `/reports/monitoring`. **Committer** : `*.dvc`,
   `dvc.lock`, `metrics.json` (out DVC `cache: false`), `reports/promotions.md`
   (journal append-only), `uv.lock`.
4. Les homes Dagster `.tmp_dagster_home_<suffixe-aléatoire>/` changent à chaque
   exécution : ne jamais les `git add`, même par wildcard.
5. Secrets uniquement via l'environnement (`.env` local issu de `.env.example`,
   jamais commité ; secrets GitHub en CI). Le hook `detect-private-key` et
   `check-added-large-files` (`--maxkb=1024`) gardent le repo sain.

## 4. Conventions de code

- `ruff` : `line-length = 100`, `target-version = "py311"`,
  `select = ["E", "W", "F", "I", "UP", "B", "SIM"]`, `src = ["src", "pipelines", "tests"]`.
- `mypy` : `disallow_untyped_defs = true`, `no_implicit_optional = true`,
  `warn_redundant_casts`, `warn_unused_ignores`, `ignore_missing_imports = true`
  (libs MLOps sans stubs : feast, dagster… ne bloquent pas).
- Chaque module a un **README court** (`src/*/README.md`, `data/README.md`,
  `docker/README.md`, `features/README.md`) : toute feature qui change un
  contrat (schéma, CLI, endpoint, métrique) met à jour le README concerné.
- `docs/decisions.md` : une ligne par ADR (contexte → décision → conséquence).
  **Les ADR acceptées sont figées** ; une révision = une nouvelle ADR.
- Messages de commit : `feat: …`, `fix(scope): …`, `docs: …` (cf. `git log`).
  **Jamais de trailer `Co-Authored-By`** ni de lien de session.
- Opérations GitHub (PR, issues, releases) : passer par le plugin GitHub
  (serveur MCP), jamais `gh` ni curl vers l'API.

## 5. Données — DVC / Great Expectations / Feast

- `dvc.yaml` : `prepare` → `validate` → `train`. `train` dépend de
  `reports/data_docs` (out de `validate`) : **l'ordre gate → training est
  structurel**, ne pas le contourner.
- Suites GE **en code** (`src/data/expectations.py`), revues en PR. Le store
  `gx/` est un cache auto-réparé (pas de dep DVC dessus) ; le rapport HTML
  `reports/data_docs/` est versionné via `dvc.lock`, même gate rouge.
- `make repro` sans changement est idempotent (`up to date`). Changer
  `train.logreg.C` ne relance que `train` (repro sélectif par section params).
- Test de blocage / recovery :
  ```bash
  .venv/bin/python -m src.data.generate_raw --drift corrupt
  make repro   # -> [PIPELINE BLOQUÉ — gate GE]
  git checkout -- data/raw.dvc dvc.lock && dvc checkout data/raw.dvc && dvc checkout
  make repro   # vert à nouveau
  ```
- Feast : `make feast-apply && make feast-materialize && make train-feast`.
  Feature view en code, source = parquet de `prepare`, équivalence CSV↔Feast
  **bit-à-bit** (testée). Registry/online store sqlite dans `data/feast/` :
  dérivé, non versionné. **DVC = vérité batch, Feast = couche de serving.**

## 6. Training / registre / promotion

- Chaque training enregistre une version de `churn-template` et pointe l'alias
  **`challenger`**. Promotion vers **`prod`** = décision séparée sous
  **double gate** : tests modèle verts (pytest en subprocess, modèle chargé
  **par alias**) + `roc_auc` strictement supérieur au champion.
- Journal `reports/promotions.md` append-only (PROMU/REFUSÉ/ROLLBACK +
  qui/quoi/quand/métriques) : il survit au serveur MLflow, ne pas le reformater
  (exclu de prettier, comme `dvc.lock` et `gx/`).
- Le Pipeline sklearn embarque le preprocessing (zéro train/serving skew) et la
  signature est validée au training.

## 7. Dagster — pièges connus (ne pas réintroduire)

- Les assets appellent **les mêmes fonctions `src/`** que `dvc.yaml` : zéro
  duplication. Toute logique dupliquée entre `pipelines/` et `src/` est un bug.
- Sélection d'assets par clés (`AssetSelection.assets`), **jamais par string** :
  dvc→omegaconf fige le runtime ANTLR à 4.9 alors que le parser de strings de
  Dagster exige 4.13.
- **Pas de `from __future__ import annotations` dans `pipelines/`** : Dagster
  valide les annotations au runtime.

## 8. Serving canary + rollback

- L'image serving **ne contient pas de modèle** : FastAPI charge `prod` au
  démarrage depuis le registre, config 100 % par env (`MODEL_NAME`,
  `MODEL_ALIAS`, `MLFLOW_TRACKING_URI` — `MODEL_NAME` doit être honoré, pas
  codé en dur). Endpoints : `/health /predict /metrics`.
- Canary nginx 90/10 (`stable :8001`, `canary :8002`, mix `:8090`). Le canary
  est le même service : dégradation simulable via `SERVE_FAILURE_CANARY=1`
  (le smoke-test doit alors échouer, ~10 % d'erreurs).
- `make smoke` : gate taux d'erreur du mix vs baseline stable (exit 1 = refus).
- `make rollback` : repointe `prod` vers la version antérieure du journal +
  recrée les conteneurs. Vérif : `curl localhost:8090/health` affiche la
  version servie.

## 9. Monitoring BP3 — règles métier

- Trafic : `/predict` loggue chaque inférence en JSONL (`data/inferences/`,
  `BackgroundTasks` non-bloquant), contrat stable avec **`prediction_id`**.
- **Prediction drift** : écart absolu de moyenne
  `|mean(churn_probability) − mean(cible train)| > 0.10` (`method="mean_gap"`).
  Ne pas réintroduire de KS continu-vs-binaire (comparer des probas continues
  à une cible binaire bruitée donne des p-values artificielles).
- **Performance différée** : jointure réelle sur `prediction_id` (Option A,
  `evaluate_performance`, champ `join` pour la traçabilité). Repli positionnel
  **uniquement** pour CSV legacy sans IDs.
- Seuils : `share_threshold: 0.3`, `pvalue_threshold: 0.05`,
  `min_current_rows/min_reference_rows: 200` (significativité KS/Chi-2).
- Retrain = drift **OU** performance dégradée (`drop_tolerance: 0.05`) **ET**
  volume (`min_new_rows: 500`) **ET** cooldown (`cooldown_hours: 24`).
  `make drift-check` : décision seule, **exit 2 = retrain requis**.
- Doc : ADR `docs/adr/0011-monitoring-and-drift.md`, READMEs `src/monitoring/`
  et `src/serving/`.

## 10. Tests

- `make test` = 55 unitaires (`tests/data`, `tests/model`, `tests/monitoring`,
  feast en repo isolé). `make test-integration` = pipeline bout-en-bout
  (dataset régénéré, `dvc repro`, tests modèle sur challenger).
- `tests/model` charge le challenger **par alias** (comme `promote`) : un test
  qui charge un chemin de modèle en dur est suspect.
- CI (`.github/workflows/ci.yml`) : lint + unitaires + intégration, **sans
  docker** (MLflow sqlite local, dataset régénéré in-process), caches uv+DVC,
  résumé métriques en PR, rapport GE en artefact si échec. CD modèle
  (`cd-model.yml`) seule à nécessiter docker, avec **rollback automatique** si
  le canary échoue.

## 11. Validation avant de considérer une tâche finie

1. `make lint && make test` verts localement.
2. Si le pipeline est touché : `make data && make repro` + `make test-integration`
   si pertinent, ou au minimum clone frais :
   ```bash
   rm -rf /tmp/check && git clone <repo> /tmp/check
   cd /tmp/check && make setup && make data && make repro && make test
   ```
   Doit réussir **sans intervention manuelle** (aucun `mlflow db upgrade`).
3. Avant commit : `git status`, `git diff`, `git log --oneline -10` ; ne stager
   que les fichiers voulus, jamais de secrets ; vérifier
   `git ls-files | grep -E "\.db$|tmp_dagster"` vide. Avant push : relire le
   diff poussé (`git diff origin/main..HEAD --stat`).
