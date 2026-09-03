# Point d'entrée unique du template : toute action passe par `make`.
.DEFAULT_GOAL := help
SHELL := /bin/bash

# Venv projet en tête de PATH pour toutes les recettes (et les sous-processus
# DVC/Dagster) : `python` dans dvc.yaml résout .venv sans activation manuelle.
export PATH := $(CURDIR)/.venv/bin:$(PATH)

# Compose : fichier dans docker/, .env optionnel (créé par `make setup`).
ENV_FILE := $(if $(wildcard .env),$(abspath .env),/dev/null)
COMPOSE := docker compose --env-file $(ENV_FILE) -f docker/compose.yaml

# Charge .env s'il existe et exporte ses variables vers les recettes
# (ex. MLFLOW_TRACKING_URI utilisée par `make health`).
ifneq (,$(wildcard .env))
include .env
export
endif

UV := $(shell command -v uv 2>/dev/null)

.PHONY: help
help: ## Affiche cette aide
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.PHONY: setup
setup: ## Crée .venv + dépendances + .env + hooks pre-commit
	@[ -f .env ] || cp .env.example .env
ifneq ($(UV),)
	uv sync --extra dev
else
	python3.11 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -e ".[dev]"
endif
	.venv/bin/pre-commit install

.PHONY: data
data: ## Génère le dataset simulé et le versionne via DVC (data/raw.dvc)
	.venv/bin/python -m src.data.generate_raw
	.venv/bin/dvc add data/raw

.PHONY: repro
repro: ## Exécute le pipeline DVC (prepare -> train)
	.venv/bin/dvc repro

.PHONY: metrics
metrics: ## Affiche les métriques DVC du dernier run
	.venv/bin/dvc metrics show

.PHONY: feast-apply
feast-apply: ## Applique les définitions Feast (entité, feature view)
	(cd features && feast apply)

.PHONY: feast-materialize
feast-materialize: ## Matérialise les features vers l'online store local
	.venv/bin/python -m src.features.store

.PHONY: train-feast
train-feast: ## Entraîne via les features servies par Feast (historical)
	.venv/bin/python -m src.training.train --source feast

.PHONY: train
train: ## Pipeline complet : stack up + dvc repro (gate GE) + log MLflow
	$(MAKE) up
	$(MAKE) repro

.PHONY: promote
promote: ## Promotion challenger->prod (gate tests modèle + métriques) + journal
	.venv/bin/python -m src.training.promote

.PHONY: rollback
rollback: ## Alias prod -> version précédente + reload du serving canary
	.venv/bin/python -m src.training.promote --rollback
	@$(COMPOSE) --profile serving up -d --force-recreate \
		serving-stable serving-canary 2>/dev/null || true

.PHONY: serve
serve: ## Serving canary local : nginx :8090 (90/10), stable :8001, canary :8002
	$(MAKE) up
ifneq ($(SERVING_IMAGE),)
	$(COMPOSE) --profile serving up -d --wait
else
	$(COMPOSE) --profile serving up -d --build --wait
endif

.PHONY: smoke
smoke: ## Smoke-test canary — exit 1 si le mix dégrade le taux d'erreur
	.venv/bin/python -m src.serving.smoke --url http://localhost:8090 --baseline http://localhost:8001

.PHONY: serve-down
serve-down: ## Arrête le serving SEULEMENT (mlflow reste up)
	$(COMPOSE) rm -sf serving-stable serving-canary nginx

.PHONY: monitoring-up
monitoring-up: ## Prometheus local :9090 (scrape /metrics du serving)
	$(COMPOSE) --profile monitoring up -d --wait prometheus

.PHONY: drift-report
drift-report: ## Rapport Evidently + résumé JSON (fenêtre d'inférences vs baseline DVC)
	.venv/bin/python -m src.monitoring.drift_detector $(ARGS)

.PHONY: drift-check
drift-check: ## Décision de réentraînement (exit 2 si retrain requis, sans déclencher)
	.venv/bin/python -m src.monitoring.retrain_policy --check; test $$? -ne 0 || true

.PHONY: retrain-if-drifted
retrain-if-drifted: ## Réentraîne (training DVC) si drift + volume + cooldown OK (cron/CI)
	.venv/bin/python -m src.monitoring.retrain_policy

.PHONY: simulate-traffic
simulate-traffic: ## Trafic simulé vers le serving (ARGS="--mode data-drift --n 500")
	.venv/bin/python -m src.monitoring.simulate_production $(ARGS)

.PHONY: monitoring-run
monitoring-run: ## Job Dagster monitoring (drift -> arbitrage -> retrain conditionnel)
	.venv/bin/dagster job execute -m pipelines.monitoring_pipeline -j monitoring_job

.PHONY: dagster-dev
dagster-dev: ## UI Dagster locale — job training_job (validate->feast->train->promote)
	.venv/bin/dagster dev -m pipelines

.PHONY: dagster-run
dagster-run: ## Exécute le job training_job en CLI (sans UI)
	.venv/bin/dagster job execute -m pipelines -j training_job

.PHONY: up
up: ## Démarre la stack locale (MLflow : tracking + registre + artefacts)
	$(COMPOSE) up -d --build --wait

.PHONY: up-postgres
up-postgres: ## Variante : MLflow sur backend postgres (profil compose)
	$(COMPOSE) --profile postgres up -d --build --wait

.PHONY: down
down: ## Stoppe la stack (les volumes sont conservés)
	$(COMPOSE) down

.PHONY: down-volumes
down-volumes: ## Stoppe la stack et supprime les volumes (reset complet)
	$(COMPOSE) down -v

.PHONY: health
health: ## Healthcheck MLflow
	curl -fsS $${MLFLOW_TRACKING_URI:-http://localhost:5000}/health

.PHONY: logs
logs: ## Suit les logs du serveur MLflow
	$(COMPOSE) logs -f mlflow

.PHONY: lint
lint: ## ruff check + ruff format --check + mypy
	.venv/bin/ruff check src pipelines tests
	.venv/bin/ruff format --check src pipelines tests
	.venv/bin/mypy

.PHONY: format
format: ## Formate et auto-fixe le code
	.venv/bin/ruff format src pipelines tests
	.venv/bin/ruff check --fix src pipelines tests

.PHONY: precommit
precommit: ## Exécute pre-commit sur tous les fichiers
	.venv/bin/pre-commit run --all-files

.PHONY: test
test: ## Tests unitaires (les tests d'intégration sont exclus par défaut)
	.venv/bin/pytest

.PHONY: test-integration
test-integration: ## Tests d'intégration (pipeline bout-en-bout)
	.venv/bin/pytest -m integration

.PHONY: clean
clean: ## Supprime caches et .venv
	rm -rf .venv .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
