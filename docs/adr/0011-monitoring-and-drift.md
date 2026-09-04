# ADR-011 : Monitoring & Drift

Date : 2026-09-03. Statut : acceptée.

## Contexte

BP1/BP2 couvrent training → serving canary. Sans monitoring, un modèle en
production se dégrade silencieusement (distribution d'entrée qui glisse,
relation features→cible qui change). BP3 ferme la boucle : inférences
loggées → drift détecté → réentraînement déclenché.

## Décisions

### 1. Evidently (présent) + tests maison (verdict), vs Alibi-Detect vs Cloud

- **Evidently** pour le rapport HTML interactif (`DataDriftPreset`) : standard
  de fait, autonome (fichier unique), intégrable sans serveur.
- **Tests KS/Chi-2 maison (numpy)** pour le verdict JSON contractuel
  (`dataset_drift`, `drift_share`) : déterministes, sans dépendance lourde,
  alignés avec Evidently sur les cas nominaux/dérivés (vérifié en tests).
- **Contre Alibi-Detect** : excellent pour la détection online (KS, MMD,
  adversarial), mais pas de rapport HTML clé-en-main — surdimensionné pour
  un template batch/quotidien.
- **Contre SageMaker Model Monitor / solutions managées** : verrouillage
  cloud, coût, non-reproductibles en local — contraires au principe
  agnostique du template. La matrice README documente la montée en charge
  (Parquet→S3, Prometheus→managed) sans changer les interfaces.

### 2. Fenêtre d'inférence : N minimal pour la significativité

`monitoring.drift.min_current_rows: 200` (et `min_reference_rows: 200`) :
sous ce volume, les tests KS/Chi-2 manquent de puissance (faux négatifs) —
le détecteur émet un `warnings` et un verdict indicatif. `retrain.min_new_rows:
500` conditionne le réentraînement : on ne réentraîne jamais sur un
échantillon non significatif. Ces seuils sont calibrés sur le dataset simulé
(8000 lignes train) et à recalibrer chez un client (règle : ≥ 200 lignes et
≥ 5 % du train, le max des deux).

### 3. Logging local JSONL partitionné vs Kafka en prod

- **Template : JSONL append-only** (`data/inferences/YYYY-MM-DD/`) : zéro
  infra, crash-safe, rejouable, requêtable via `latest_inference_frames`.
  Écriture via `BackgroundTasks` (latence /predict inchangée).
- **Prod : Parquet partitionné (S3/MinIO) puis Kafka si besoin** : le contrat
  (une ligne typée par prédiction, `prediction_id` comme clé) ne change pas —
  seul le sink change. Kafka n'est justifié qu'au-delà de ~1000 req/s ou pour
  du scoring temps réel multi-consommateurs ; en deçà, le batch JSONL→Parquet
  quotidien suffit et coûte un ordre de grandeur moins cher.

### 4. Réentraînement : drift OU dégradation, avec cooldown

`should_retrain = (dataset_drift OU performance_dégradée) ET volume ET
cooldown(24h)`. Le cooldown file-based (`retrain_state.json`) évite les
boucles de retrain sur drift persistant. Le retrain passe par le training
DVC existant (mêmes données + nouvelles inférences labellisées côté client).

## Conséquences

- Nouvelles deps : `evidently` (extra `monitoring`), `prometheus-client`
  (extra `serving`, déjà transitive). Image serving reconstruite.
- `/metrics` non routé par nginx (404 edge) — scrape interne uniquement.
- `reports/monitoring/` et `data/inferences/` ignorés par git (artefacts).
- Sensor Dagster `drift_sensor` (30 min) + job `monitoring_job` ; CLI
  `make retrain-if-drifted` pour cron/CI.
