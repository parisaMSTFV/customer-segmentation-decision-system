from __future__ import annotations

from pathlib import Path

import pytest

from customer_segmentation.pipeline import run_pipeline


@pytest.fixture(scope="session")
def pipeline_output(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, object]]:
    output_root = tmp_path_factory.mktemp("segmentation-pipeline")
    return output_root, run_pipeline(output_root)
