# features/ — repo Feast (provider local/file)

- `feature_store.yaml` : config du repo — registry et online store (sqlite) dans `data/feast/`, **artefacts dérivés** du materialize (ni git, ni DVC).
- `feature_views.py` : entité `customer`, source = parquet préparé par l'étape DVC `prepare`, feature view `customer_profile` (mêmes features que `params.yaml`).

## Commandes

```bash
make feast-apply         # (ré)enregistre les définitions (registry.db)
make feast-materialize   # online store : materialize du dataset complet (idempotent)
make train-feast         # training via get_historical_features (métriques identiques au chemin DVC)
```

## Règle du template

Le feature view expose EXACTEMENT les features du training DVC. L'équivalence
des deux chemins est testée (`tests/data/test_feast.py`) — c'est la garantie
anti skew train/serving. Toute divergence = test rouge en PR.
