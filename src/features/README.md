# src/features — couche Feast

- `store.py` : accès au feature store — `materialize_latest` (online store local), `load_training_frame` (offline, point-in-time correct), `get_store`.
- Le chemin **batch de référence reste DVC** (data/prepared) ; Feast ajoute le point-in-time et le serving temps réel (`get_online_features`, étape 10).
- Définitions (entité, source, feature view) : `features/feature_views.py`.
