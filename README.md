# mlops-template

Template MLOps réutilisable, agnostique cloud, conçu pour être adapté par client.

- **BP1 — training reproductible** : DVC (données), Feast (features), MLflow (expériences/registre), Dagster (orchestration).
- **BP2 — CI/CD modèle** : Great Expectations en gate bloquante, tests de modèle (métriques/invariance/comportement), promotion par alias MLflow, canary + rollback.

Le repo se construit pas à pas : chaque étape est un commit autonome, chaque module a son README court. L'historique git est le support de formation.

## Quickstart

```bash
make setup    # .venv (uv), dépendances, .env, hooks pre-commit
make up       # stack docker locale (MLflow sur :5000)
make health   # {"status": "ok"} attendu
```

## Structure

```
mlops-template/
├── data/                  # données versionnées via DVC (dvc.lock commité)
├── features/              # repo Feast (feature_store.yaml, entities, feature views)
├── src/
│   ├── data/              # ingestion, validation GE, préparation
│   ├── features/          # définitions Feast + materialize
│   ├── training/          # entraînement, évaluation, packaging modèle
│   ├── serving/           # API FastAPI + chargement modèle depuis registre
│   └── monitoring/        # hooks (BP3 — emplacement réservé)
├── pipelines/             # jobs/assets Dagster
├── tests/                 # data/ model/ integration/
├── .github/workflows/     # ci.yml, cd-model.yml
├── docker/                # Dockerfiles + compose (MLflow, postgres optionnel)
├── Makefile               # point d'entrée unique
├── dvc.yaml               # prepare -> validate -> train -> evaluate
├── params.yaml            # hyperparamètres + chemins (lus par DVC et Dagster)
└── docs/                  # ADR (docs/decisions.md)
```

## Commandes principales

| Commande                              | Rôle                                       |
| ------------------------------------- | ------------------------------------------ |
| `make setup`                          | venv uv + deps + `.env` + hooks pre-commit |
| `make up` / `make down`               | stack locale docker (MLflow)               |
| `make health`                         | healthcheck MLflow                         |
| `make lint` / `make format`           | ruff + mypy                                |
| `make precommit`                      | pre-commit sur tout le repo                |
| `make test` / `make test-integration` | pytest (unitaires / bout-en-bout)          |

`make help` liste tout.

## Pipeline DVC (étape 2)

```bash
make data      # génère le dataset de churn simulé (déterministe) + dvc add data/raw
make repro     # dvc repro : prepare -> train (dvc.lock tracé, métriques dans metrics.json)
make metrics   # dvc metrics show
```

**Réactivité aux params** (testée, à reproduire chez un client) : `params.yaml` est en
dépendance des étapes DVC — chaque étape ne relance que si SA section change.

- `train.logreg.C: 1.0 -> 0.1` puis `make repro` : `prepare` skippé, seul `train`
  ré-exécuté (accuracy 0.7485 -> 0.7500 dans `dvc metrics diff`) ;
- retour à `C: 1.0` puis `make repro` : `Stage 'train' is cached` — les sorties
  exactes du premier run sont restituées depuis le cache DVC ;
- `make repro` sans changement : `Data and pipelines are up to date.` (idempotent).

Le contenu de `data/` vit dans le cache DVC ; git ne suit que les pointeurs
(`data/raw.dvc`, `dvc.lock`). Remote distant (S3/MinIO) : cf. `data/README.md`.

## Gate données — Great Expectations (étape 3)

`make repro` enchaîne prepare -> **validate** -> train. Le train ne s'exécute
que si la validation passe : le DAG DVC rend l'ordre structurel (`train` dépend
de `reports/`, l'out de `validate`).

- Suites définies **en code** (`src/data/expectations.py`) : schéma exact, plages,
  ensembles, distribution clé (taux de churn) — revues en PR comme du code.
- Rapport HTML **versionné** : `reports/data_docs/` est un out DVC (tracé dans
  `dvc.lock`), publié même quand la gate est rouge.
- Test de blocage (à refaire chez un client) :

  ```bash
  .venv/bin/python -m src.data.generate_raw --drift corrupt   # nulls + hors-plage + catégorie inconnue
  make repro        # -> [PIPELINE BLOQUÉ — gate GE] : détail par dataset et par colonne
  # recovery (dvc repro en échec réécrit le pointeur du dataset modifié) :
  git checkout -- data/raw.dvc dvc.lock && dvc checkout data/raw.dvc && dvc checkout
  make repro        # vert à nouveau
  ```

- Le drift de distribution (`--drift shift`) est bloqué aussi : l'expectation de
  moyenne sur la cible (`churn`) sort de ses bornes.

## Feature store — Feast (étape 4)

```bash
make feast-apply && make feast-materialize && make train-feast
```

Le feature view `customer_profile` expose exactement les features du training
DVC, depuis le parquet produit par `prepare`. `make train-feast` fit le même
pipeline via `get_historical_features` : métriques **bit-à-bit identiques**
au chemin fichiers (testé aussi en unitaire, repo Feast isolé).

### DVC vs Feast — quand utiliser quoi

| Situation                                                     | Choix                                        |
| ------------------------------------------------------------- | -------------------------------------------- |
| Training batch reproductible, une seule source                | CSV/parquet DVC (chemin par défaut du train) |
| Features partagées training ↔ serving temps réel             | Feast (`get_online_features`, étape 10)      |
| Historique long + features calculées à la volée multi-équipes | Feast offline store (point-in-time correct)  |
| Données qui changent sans retraining                          | Feast (materialize régulier) > rebuild DVC   |

Dans ce template : **DVC = source de vérité batch, Feast = couche de serving**
de ces mêmes features. Un client "batch pur" peut désactiver Feast sans
toucher au training ; un client temps réel garde les deux, testés cohérents.

## État d'avancement

- [x] **Étape 1** — squelette reproductible (uv 3.11, pre-commit, compose MLflow, Makefile)
- [x] **Étape 2** — données versionnées DVC (dataset simulé, prepare/train, params.yaml)
- [x] **Étape 3** — gate GE bloquante (suites en code, rapport HTML versionné, blocage testé)
- [x] **Étape 4** — Feast local (feature view, materialize, chemin training équivalent)
- [ ] Étapes 5-8 — training/MLflow, tests modèle, Dagster, promotion
- [ ] Étapes 9-11 — CI, CD/canary/rollback, documentation
