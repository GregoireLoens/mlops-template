# ADR-012 : Outs DVC déterministes (sentinelle GE, model card sans horloge)

Date : 2026-09-04. Statut : acceptée.

## Contexte

`git status` affichait `M dvc.lock` après chaque `make repro` sans aucun
changement de code : deux outs DVC embarquaient du temps mur-horloge
(`models/model_card.md`, `reports/data_docs` avec ses dossiers de validation
horodatés par Great Expectations). Le lock n'était plus l'état figé du code.

## Décisions

- `model_card.md` sans timestamp : provenance via le hash court du dataset
  brut (déterministe) + la fenêtre temporelle déjà présente.
- `reports/data_docs` retiré des outs DVC (Option B) : GE horodate ses
  chemins ET ses pages HTML — un out byte-identique exigerait de figer
  l'horloge, contre l'outil. Le rapport reste généré localement (diagnostic
  - artefact CI), mais la gate devient structurelle via la sentinelle
    `reports/validate.ok` (volumes validés, écrite uniquement si gate verte),
    en dépendance de `train`. Réviser ADR-003 en ce sens (rapport non versionné).

## Conséquences

`make repro` sans changement est idempotent (`up to date`, lock stable).
Sur données corrompues : `validate` échoue toujours (pas de sentinelle) et le
pipeline s'arrête avant `train` — la gate ne dépend pas du rapport HTML.
