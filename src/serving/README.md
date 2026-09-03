# src/serving — API de serving + gate canary + monitoring (BP3)

- `app.py` : FastAPI. `/health` (statut + version servie), `/predict`
  (features brutes -> probabilité de churn + `prediction_id`), `/reload`
  (recharge l'alias), `/metrics` (Prometheus — NON routé par nginx).
  Chargement du modèle au démarrage **par alias** (`prod` par défaut) depuis
  le registre MLflow — jamais par chemin disque, jamais baked dans l'image.
  Pydantic valide bornes et catégories : une requête invalide est rejetée
  (422) au lieu de produire silencieusement un vecteur de features vide.
- `inference_log.py` : logging JSONL horodaté (`data/inferences/YYYY-MM-DD/`),
  via `BackgroundTasks` (non-bloquant). Interrupteur `LOG_INFERENCES`,
  répertoire `INFERENCES_DIR`. Le `prediction_id` renvoyé = clé de jointure
  avec la vérité terrain retardée.
- `metrics.py` : compteurs requêtes, histogramme latences, gauges
  `data_drift_share` / `prediction_drift_share` (depuis le dernier
  `drift_summary.json`), taux d'erreur 5xx.
- `smoke.py` : gate canary — N requêtes via l'edge (nginx 90/10) comparées à
  la baseline stable jointe directement. Taux d'erreur au-dessus de la
  marge -> exit 1 -> le CD joue le rollback.

Config par env uniquement : `MODEL_NAME`, `MODEL_ALIAS`,
`MLFLOW_TRACKING_URI`, `SERVE_FAILURE=1` (simulation d'échec canary),
`LOG_INFERENCES`, `INFERENCES_DIR`.

Tester en local :

```bash
make serve && make smoke
curl -s localhost:8090/health
curl -s -X POST localhost:8090/predict -H "Content-Type: application/json" \
  -d '{"age":35,"tenure_months":12,"monthly_fee":59.9,"num_support_calls":3,"has_premium":0,"contract_type":"month_to_month","signup_channel":"web"}'
```
