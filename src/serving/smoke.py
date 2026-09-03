"""Smoke-test du routing canary : compare le taux d'erreur de l'edge (nginx
90/10) à la baseline stable directement.

N charge le mix traffic : si le canary échoue (ex. model cassé), ~10 % des
appels passant par nginx tombent en 5xx et le taux d'erreur global dépasse
la marge au-dessus de la baseline -> exit 1 (le CD déclenche le rollback).

Usage : python -m src.serving.smoke --url http://localhost:8080 \
        --baseline http://localhost:8001 --n 100
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request

# Payload valide pour le modèle (bornes = Literal/pydantic de src/serving/app).
PAYLOAD = {
    "age": 35,
    "tenure_months": 12.0,
    "monthly_fee": 59.9,
    "num_support_calls": 3,
    "has_premium": 0,
    "contract_type": "month_to_month",
    "signup_channel": "web",
}


def _error_rate(url: str, n: int) -> float:
    errors = 0
    for _ in range(n):
        req = urllib.request.Request(
            url + "/predict",
            data=json.dumps(PAYLOAD).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                resp.read()
        except urllib.error.HTTPError as e:
            if e.code >= 500:
                errors += 1
            else:  # 4xx = mauvais payload (faute du test, pas du service)
                raise
        except urllib.error.URLError:
            errors += 1
    return errors / n


def run(url: str, baseline: str, n: int, margin: float) -> bool:
    base_rate = _error_rate(baseline, n)
    route_rate = _error_rate(url, n)
    ok = route_rate <= base_rate + margin
    print(
        f"[smoke] baseline={base_rate:.0%} canary-mix={route_rate:.0%} "
        f"(marge {margin:.0%}) -> {'OK' if ok else 'ÉCHEC — rollback requis'}"
    )
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8080", help="route canary (nginx)")
    parser.add_argument("--baseline", default="http://localhost:8001", help="stable direct")
    parser.add_argument("--n", type=int, default=100, help="nombre de requêtes par cible")
    parser.add_argument("--margin", type=float, default=0.03, help="marge d'erreur tolérée")
    args = parser.parse_args()
    if not run(args.url, args.baseline, args.n, args.margin):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
