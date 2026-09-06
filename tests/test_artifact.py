from __future__ import annotations

import json

import pytest

from customer_segmentation.artifact import (
    MANIFEST_FILENAME,
    build_definition_payload,
    load_model_artifact,
    save_model_artifact,
)
from customer_segmentation.features import prepare_features
from customer_segmentation.modeling import build_model
from customer_segmentation.synthetic import generate_customers


def _artifact(tmp_path):
    observations, _truth = generate_customers(120, 7)
    model = build_model(3, 7).fit(prepare_features(observations))
    names = {0: "A", 1: "B", 2: "C"}
    payload = build_definition_payload(model, names, "2026-01-01", "abc", "3.0", {})
    save_model_artifact(
        tmp_path,
        model,
        names,
        payload,
        {"input_contract_version": "3.0"},
    )
    return tmp_path / MANIFEST_FILENAME


def test_artifact_loads_only_matching_definition_and_runtime(tmp_path) -> None:
    manifest_path = _artifact(tmp_path)
    model, names, manifest = load_model_artifact(tmp_path)
    assert model.named_steps["cluster"].n_clusters == 3
    assert names == {0: "A", 1: "B", 2: "C"}
    assert manifest["runtime_versions"]["customer_segmentation"] == "2.1.0"

    changed = json.loads(manifest_path.read_text(encoding="utf-8"))
    changed["runtime_versions"]["scikit_learn"] = "0.0"
    manifest_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="runtime versions"):
        load_model_artifact(tmp_path)


def test_artifact_rejects_model_path_and_definition_tampering(tmp_path) -> None:
    manifest_path = _artifact(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["model_file"] = "../unsafe.joblib"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported model path"):
        load_model_artifact(tmp_path)

    manifest_path = _artifact(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["definition_payload"]["snapshot_date"] = "2027-01-01"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="definition integrity"):
        load_model_artifact(tmp_path)
