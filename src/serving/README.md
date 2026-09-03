# src/serving — API de serving + gate canary

- `app.py` : FastAPI. `/health` (statut + version servie), `/predict`
  (features brutes -> probabilité de churn), `/reload` (recharge l'alias).
  Chargement du modèle au démarrage **par alias** (`prod` par défaut) depuis
  le registre MLflow — jamais par chemin disque, jamais baked dans l'image.
  Pydantic valide bornes et catégories : une requête invalide est rejetée
  (422) au lieu de produire silencieusement un vecteur de features vide.
- `smoke.py` : gate canary — N requêtes via l'edge (nginx 90/10) comparées à
  la baseline stable jointe directement. Taux d'erreur au-dessus de la
  marge -> exit 1 -> le CD joue le rollback.

Config par env uniquement : `MODEL_NAME`, `MODEL_ALIAS`,
`MLFLOW_TRACKING_URI`, `SERVE_FAILURE=1` (simulation d'échec canary).

Tester en local :

```bash
make serve && make smoke
curl -s localhost:8090/health
curl -s -X POST localhost:8090/predict -H "Content-Type: application/json" \
  -d '{"age":35,"tenure_months":12,"monthly_fee":59.9,"num_support_calls":3,"has_premium":0,"contract_type":"month_to_month","signup_channel":"web"}'
```
