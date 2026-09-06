from __future__ import annotations

import json
from pathlib import Path

import pytest

from customer_segmentation.config import load_config

ROOT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "analysis.json"


def test_bundled_and_repository_configs_match() -> None:
    assert load_config() == load_config(ROOT_CONFIG)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda raw: raw.update(candidate_clusters=[3, 3]), "unique"),
        (lambda raw: raw.update(selection_seeds=[11, 11]), "unique"),
        (lambda raw: raw.update(minimum_silhouette=2), "minimum_silhouette"),
        (lambda raw: raw.update(minimum_resample_stability=-0.1), "stability"),
        (
            lambda raw: raw.update(selection_weights={"silhouette": -1, "stability": 2}),
            "nonnegative",
        ),
    ],
)
def test_invalid_configs_fail_closed(tmp_path, change, message: str) -> None:
    raw = json.loads(ROOT_CONFIG.read_text(encoding="utf-8"))
    change(raw)
    path = tmp_path / "analysis.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_config(path)
