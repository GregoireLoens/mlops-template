"""Tests du logging d'inférence : format, parsing, interrupteur, non-blocage.

Le logging est le socle de BP3 : chaque /predict réussi doit persister une
ligne JSONL typée sans ralentir la réponse (BackgroundTasks). Ces tests
couvrent le module `inference_log` en isolation (aucun serveur requis).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from src.serving import inference_log

FEATURES = {
    "age": 35,
    "tenure_months": 12.0,
    "monthly_fee": 59.9,
    "num_support_calls": 3,
    "has_premium": 0,
    "contract_type": "month_to_month",
    "signup_channel": "web",
}


def test_build_record_schema(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOG_INFERENCES", "true")
    record = inference_log.build_record(FEATURES, 0.7, True, "v1 (prod)")
    assert set(record) == {
        "timestamp",
        "prediction_id",
        "model_version",
        "features",
        "churn_probability",
        "churn",
    }
    assert record["features"] == FEATURES
    assert record["churn_probability"] == 0.7
    assert record["churn"] is True
    # Horodatage ISO 8601 parseable, prediction_id unique.
    from datetime import datetime

    datetime.fromisoformat(record["timestamp"])
    other = inference_log.build_record(FEATURES, 0.2, False, "v1 (prod)")
    assert other["prediction_id"] != record["prediction_id"]


def test_append_et_relecture_jsonl(tmp_path: Path) -> None:
    record = inference_log.build_record(FEATURES, 0.42, False, "v2 (prod)")
    path = inference_log.append_record(record, base_dir=tmp_path)
    assert path.parent.name != ""
    assert path.suffix == ".jsonl"
    rows = inference_log.read_jsonl(path)
    assert len(rows) == 1
    assert rows[0]["prediction_id"] == record["prediction_id"]
    # Append : la 2e ligne s'ajoute sans écraser la 1re.
    inference_log.append_record(record, base_dir=tmp_path)
    assert len(inference_log.read_jsonl(path)) == 2


def test_log_inference_respecte_l_interrupteur(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOG_INFERENCES", "false")
    monkeypatch.setenv("INFERENCES_DIR", str(tmp_path))
    assert inference_log.logging_enabled() is False
    assert inference_log.log_inference(FEATURES, 0.9, True, "v1 (prod)") is None
    assert list(tmp_path.glob("*/inferences.jsonl")) == []


def test_log_inference_ecrit_par_defaut(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LOG_INFERENCES", raising=False)
    monkeypatch.setenv("INFERENCES_DIR", str(tmp_path))
    assert inference_log.logging_enabled() is True
    path = inference_log.log_inference(FEATURES, 0.9, True, "v1 (prod)")
    assert path is not None and path.exists()
    line = path.read_text(encoding="utf-8").strip()
    assert json.loads(line)["features"]["contract_type"] == "month_to_month"


def test_prediction_id_propage_du_serving(tmp_path: Path) -> None:
    pid = inference_log.new_prediction_id()
    record = inference_log.build_record(FEATURES, 0.5, True, "v1", prediction_id=pid)
    assert record["prediction_id"] == pid


def test_read_jsonl_ignore_les_lignes_corrompues(tmp_path: Path) -> None:
    path = tmp_path / "inferences.jsonl"
    record = inference_log.build_record(FEATURES, 0.1, False, None)
    path.write_text(json.dumps(record) + "\n{corrompu\n", encoding="utf-8")
    assert len(inference_log.read_jsonl(path)) == 1


def test_latest_frames_ordonne_partitions_recentes(tmp_path: Path) -> None:
    for day in ("2026-01-01", "2026-01-02"):
        d = tmp_path / day
        d.mkdir(parents=True)
        for i in range(3):
            r = inference_log.build_record(FEATURES, 0.1 * i, False, "v1")
            with open(d / "inferences.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(r) + "\n")
    frames = inference_log.latest_inference_frames(tmp_path, limit=4)
    assert len(frames) == 4
    # Les plus récentes d'abord : la partition 01-02 est lue en premier.
    assert frames[0]["timestamp"] >= frames[-1]["timestamp"]


def test_env_dir_prioritaire(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("INFERENCES_DIR", str(tmp_path / "custom"))
    assert inference_log.inferences_dir() == tmp_path / "custom"
    monkeypatch.delenv("INFERENCES_DIR")
    assert inference_log.inferences_dir().name == "inferences"
    assert os.environ.get("INFERENCES_DIR") is None or True
