from __future__ import annotations

import json

import pandas as pd
import pytest

from customer_segmentation.external import run_external_segmentation
from customer_segmentation.synthetic import generate_customers


def test_external_segmentation_writes_stable_decision_schema(tmp_path) -> None:
    observations, _truth = generate_customers(900, 17)
    input_path = tmp_path / "customer_features.csv"
    output_dir = tmp_path / "output"
    observations.to_csv(input_path, index=False)

    metrics = run_external_segmentation(input_path, output_dir, seed=17)
    assignments = pd.read_csv(output_dir / "customer_segments.csv")
    metadata = json.loads((output_dir / "run_metadata.json").read_text(encoding="utf-8"))

    assert assignments.columns.tolist() == [
        "customer_id",
        "cluster_id",
        "segment_name",
        "segment_definition_id",
        "assignment_status",
    ]
    assert len(assignments) == 900
    assert assignments["segment_name"].nunique() == 6
    assert assignments["segment_definition_id"].nunique() == 1
    assert not {"synthetic_persona", "segment", "label"}.intersection(assignments)
    assert metrics == metadata
    assert metadata["campaign_impact"] == "Not evaluated"
    assert (output_dir / "segment_profiles.csv").is_file()
    assert (output_dir / "decision_playbook.csv").is_file()


def test_external_segmentation_rejects_too_few_customers(tmp_path) -> None:
    observations, _truth = generate_customers(700, 17)
    input_path = tmp_path / "too_small.csv"
    observations.head(59).to_csv(input_path, index=False)

    with pytest.raises(ValueError, match="at least 60 customers"):
        run_external_segmentation(input_path, tmp_path / "output", seed=17)
