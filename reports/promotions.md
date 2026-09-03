# Journal des promotions

Une ligne par décision : qui, quoi, quand, quelles métriques.
`prod` = alias champion (servi en production), `challenger` = dernier entraîné.

| Date (UTC)       | Qui | Modèle         | Challenger          | Champion              | Décision | Motif                                                          |
| ---------------- | --- | -------------- | ------------------- | --------------------- | -------- | -------------------------------------------------------------- |
| 2026-09-03 15:45 | glo | churn-template | v7 (roc_auc=0.7332) | — (roc_auc=—)         | PROMU    | première promotion (pas de champion)                           |
| 2026-09-03 15:45 | glo | churn-template | v7 (roc_auc=0.7332) | prod (roc_auc=0.7332) | REFUSÉ   | pas d'amélioration mesurable (roc_auc challenger <= champion)  |
| 2026-09-03 15:46 | glo | churn-template | v8 (roc_auc=0.7040) | prod (roc_auc=0.7332) | REFUSÉ   | tests modèle en échec (1 failed, 5 passed, 4 skipped in 3.10s) |
| 2026-09-03 15:57 | glo | churn-template | v9 (roc_auc=0.7332) | prod (roc_auc=0.7332) | REFUSÉ | pas d'amélioration mesurable (roc_auc challenger <= champion) |
| 2026-09-03 15:58 | glo | churn-template | v10 (roc_auc=0.7332) | prod (roc_auc=0.7332) | REFUSÉ | pas d'amélioration mesurable (roc_auc challenger <= champion) |
