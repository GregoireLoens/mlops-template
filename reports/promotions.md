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
| 2026-09-03 16:28 | glo | churn-template | v12 (roc_auc=0.7533) | prod (roc_auc=0.7332) | PROMU | roc_auc challenger > champion |
| 2026-09-03 16:28 | glo | churn-template | — | v7 | ROLLBACK | prod repointé de v12 vers v7 |
| 2026-09-03 18:17 | glo | churn-template | v13 (roc_auc=0.7332) | prod (roc_auc=0.7332) | REFUSÉ | pas d'amélioration mesurable (roc_auc challenger <= champion) |
| 2026-09-03 18:21 | glo | churn-template | — | v12 | ROLLBACK | prod repointé de v7 vers v12 |
| 2026-09-03 18:23 | glo | churn-template | v13 (roc_auc=0.7332) | prod (roc_auc=0.7533) | REFUSÉ | pas d'amélioration mesurable (roc_auc challenger <= champion) |
| 2026-09-03 18:24 | glo | churn-template | v13 (logreg-20260903-181731, roc_auc=0.7332) | prod (prod=v12, roc_auc=0.7533) | REFUSÉ | pas d'amélioration mesurable (roc_auc challenger <= champion) |
