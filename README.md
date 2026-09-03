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

## État d'avancement

- [x] **Étape 1** — squelette reproductible (uv 3.11, pre-commit, compose MLflow, Makefile)
- [ ] Étape 2 — données versionnées DVC · Étape 3 — Great Expectations · Étape 4 — Feast
- [ ] Étapes 5-8 — training/MLflow, tests modèle, Dagster, promotion
- [ ] Étapes 9-11 — CI, CD/canary/rollback, documentation
