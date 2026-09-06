.PHONY: install reproduce smoke fit-fixture score-fixture test lint security check

install:
	uv sync --locked --extra dev

reproduce:
	MPLCONFIGDIR=.matplotlib uv run --locked customer-segmentation reproduce

smoke:
	MPLCONFIGDIR=.matplotlib uv run --locked customer-segmentation smoke

fit-fixture:
	MPLCONFIGDIR=.matplotlib uv run --locked customer-segmentation fit --input data/synthetic/customer_features.csv --model-dir artifacts/model --output-dir artifacts/fit-fixture --allow-raw-identifiers

score-fixture: fit-fixture
	MPLCONFIGDIR=.matplotlib uv run --locked customer-segmentation score --input data/synthetic/customer_features.csv --model-dir artifacts/model --output-dir artifacts/score-fixture --previous-assignments artifacts/fit-fixture/customer_segments.csv --allow-raw-identifiers

test:
	MPLCONFIGDIR=.matplotlib uv run --locked pytest

lint:
	uv run --locked ruff check .
	uv run --locked ruff format --check .

security:
	uv run --locked python scripts/check_sensitive.py

check: lint test security smoke score-fixture
