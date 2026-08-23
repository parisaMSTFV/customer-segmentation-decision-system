.PHONY: install reproduce smoke fit-fixture score-fixture test lint security check

install:
	python -m pip install -e ".[dev]"

reproduce:
	MPLCONFIGDIR=.matplotlib python -m customer_segmentation.cli reproduce

smoke:
	MPLCONFIGDIR=.matplotlib python -m customer_segmentation.cli smoke

fit-fixture:
	MPLCONFIGDIR=.matplotlib python -m customer_segmentation.cli fit --input data/synthetic/customer_features.csv --model-dir artifacts/model --output-dir artifacts/fit-fixture

score-fixture: fit-fixture
	MPLCONFIGDIR=.matplotlib python -m customer_segmentation.cli score --input data/synthetic/customer_features.csv --model-dir artifacts/model --output-dir artifacts/score-fixture --previous-assignments artifacts/fit-fixture/customer_segments.csv

test:
	MPLCONFIGDIR=.matplotlib python -m pytest

lint:
	python -m ruff check .
	python -m ruff format --check .

security:
	python scripts/check_sensitive.py

check: lint test security smoke score-fixture
