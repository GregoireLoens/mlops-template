# ADR-013 : Matrice de confusion hors outs DVC (rendu police système)

Date : 2026-09-04. Statut : acceptée.

## Contexte

`git status` affichait `M dvc.lock` après un repro sans changement de code,
sur une autre machine : `models/confusion_matrix.png` (out DVC de `train`)
est rendu par matplotlib avec la police DejaVu Sans du système, qui varie
d'une machine à l'autre. Le `.pkl` et le `model_card.md` sont déterministes.

## Décisions

- Option B (retenue) : le PNG sort des outs — `train` versionne les fichiers
  `models/model.pkl` + `models/model_card.md` au lieu du dossier `models`.
  Même principe qu'ADR-012 : un diagnostic que l'outil ne peut pas stabiliser
  ne fige pas le lock. L'option A (forcer les polices embarquées) est écartée :
  l'octet-identique dépendrait aussi de freetype/fontconfig — garantie
  empirique, pas structurelle.
- Le PNG reste généré localement et loggé comme artefact MLflow (lecture
  humaine) ; aucun consommateur aval ne le lit (ni tests, ni serving).

## Conséquences

Le hash suivi par DVC ne peut plus varier avec les polices du système.
`src/training/reporting.py` rejoint les deps de `train` (il produit les outs).
