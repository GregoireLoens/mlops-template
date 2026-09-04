# Décisions d'architecture (ADR)

Une ligne par décision : contexte -> décision -> conséquence. Les ADR sont
figées une fois acceptées ; une révision se fait par une nouvelle ADR.

## ADR-001 — uv + hatchling + layout `src`

uv pour la reproductibilité du lock (uv.lock commité, install rapide en CI),
hatchling pour le packaging, layout `src` pour interdire d'importer le
package sans l'installer. Python 3.11 pin via `.python-version`.

## ADR-002 — `params.yaml` source de vérité unique

Chemin, split, features, hyperparamètres, seuils d'éval : tout dans
`params.yaml`, en dépendance DVC (repro sélectif), lu par Dagster, les tests
et la promotion. Conséquence : un changement de param ne peut pas passer
inaperçu dans le DAG.

## ADR-003 — Gate Great Expectations dans le DAG DVC

`validate` est une étape DVC dont l'out (`reports/data_docs`) est une dép
de `train` : l'ordre gate -> training est structurel, pas disciplinaire.
Le store GX (`gx/`) est un cache idempotent re-générable (pas de dép DVC
dessus — un store supprimé se reconstruit seul). Le rapport HTML est
versionné (`dvc.lock`) même quand la gate est rouge.

## ADR-004 — Feast file provider, DVC et Feast ne se chevauchent pas

DVC versionne les fichiers d'entraînement ; Feast expose ces features au
serving temps réel. Feature view défini en code, source = parquet produit
par `prepare`, équivalence CSV↔Feast testée bit-à-bit. Le registry/online
store sqlite dans `data/feast/` est dérivé et non versionné.

## ADR-005 — MLflow sqlite + serve-artifacts, aliases pas de stages

Le serveur local tourne exactement comme un déploiement scalable
(`--serve-artifacts` : les clients ne touchent jamais le disque du serveur) ;
postgres est un profil compose, pas un fork du template. Les versions sont
adressées par **alias** (`challenger`, `prod`) — les stages MLflow sont
dépréciés et globaux. Sans `MLFLOW_TRACKING_URI`, store fichier `./mlruns` :
le pipeline reste reproductible sans serveur (utilisé en CI).

## ADR-006 — Promotion : double gate + journal git commité

`make promote` exécute les tests modèle (subprocess pytest, ils chargent le
challenger PAR ALIAS) puis compare roc_auc challenger vs champion
(strictement supérieur — pas d'amélioration mesurable, pas de promotion).
La décision (PROMU/REFUSÉ/ROLLBACK, avec qui/quoi/quand/métriques) est
append-only dans `reports/promotions.md` commité : l'audit trail survit au
serveur MLflow et se relit en PR.

## ADR-007 — Dagster réutilise `src/`, sélection par clés

Un seul implémentation des étapes (fonctions `src/`), appelée à la fois par
`dvc.yaml` et par les assets Dagster : zéro duplication, mêmes gates.
Contraintes encodées dans le code : sélection d'assets par
`AssetSelection.assets` (jamais une string — le parser ANTLR de Dagster
exige un runtime 4.13 alors que dvc->omegaconf fige 4.9) et pas de
`from __future__ import annotations` dans `pipelines/` (Dagster valide les
annotations au runtime).

## ADR-008 — Serving par alias, image sans modèle

L'image serving ne contient pas de modèle : FastAPI charge `prod` au
démarrage depuis le registre (config 100% par env : `MODEL_NAME`,
`MODEL_ALIAS`, `MLFLOW_TRACKING_URI`). Conséquences : déployer = publier une
image une fois ; changer de modèle = repointer un alias ; zéro
train/serving skew (le Pipeline sklearn embarque le préprocessing).

## ADR-009 — Canary nginx par poids + smoke baseline, rollback simple

90/10 par poids nginx (équilibre statistique, suffisant pour une gate à
marge 3%). Le smoke-test compare le taux d'erreur du mix à la baseline
stable jointe directement : pas d'alerte à configurer, exit code = verdict.
Le rollback = repointer l'alias `prod` vers la version antérieure du journal

- recréer les conteneurs : deux commandes, rejouables, sans runbook.

## ADR-010 — CI sans docker, CD avec docker

La CI (lint, unitaires, intégration) tourne sans aucun conteneur : MLflow
sqlite local, dataset régénéré in-process, registre local. La CD (image
serving) est le seul workflow qui nécessite docker. Conséquence : la CI
reste rapide et portable, le coût docker n'est payé que quand on déploie.

## ADR-011 : Monitoring & Drift

Monitoring production via Evidently + tests KS/Chi-2 maison, inférences JSONL jointes par `prediction_id`, réentraînement si drift OU performance dégradée (avec volume + cooldown).
