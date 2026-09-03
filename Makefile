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
