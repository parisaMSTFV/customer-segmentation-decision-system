.PHONY: install reproduce smoke segment-fixture test lint security check

install:
	python -m pip install -e ".[dev]"

reproduce:
	MPLCONFIGDIR=.matplotlib python -m customer_segmentation.cli reproduce

smoke:
	MPLCONFIGDIR=.matplotlib python -m customer_segmentation.cli smoke

segment-fixture:
	MPLCONFIGDIR=.matplotlib python -m customer_segmentation.cli segment --input data/synthetic/customer_features.csv --output-dir artifacts/segment-fixture

test:
	MPLCONFIGDIR=.matplotlib python -m pytest

lint:
	python -m ruff check .
	python -m ruff format --check .

security:
	python scripts/check_sensitive.py

check: lint test security smoke segment-fixture
