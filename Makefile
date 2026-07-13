.PHONY: setup data dashboard features feature-check test lint format check clean

setup:
	uv sync

data:
	uv run python scripts/download_data.py --config configs/default_universe.yaml --out data/processed/adjusted_close.parquet

dashboard:
	PYTHONPATH=src uv run streamlit run app/streamlit_app.py

features:
	uv run python -m mrrp.features.build_features --feature-config configs/regime_features.yaml

feature-check:
	uv run python -m mrrp.features.validate_features

test:
	uv run pytest

lint:
	uv run ruff check .

format:
	uv run ruff format .

check:
	uv run ruff check .
	uv run ruff format --check .
	uv run pytest

clean:
	rm -rf .pytest_cache .ruff_cache
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
