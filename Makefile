.PHONY: setup data dashboard features feature-check models model-check stress backtest report test lint format check clean

setup:
	uv sync

data:
	uv run python scripts/download_data.py --config configs/default_universe.yaml --out data/processed/adjusted_close.parquet

dashboard:
	PYTHONPATH=src uv run streamlit run app/streamlit_app.py

features:
	PYTHONPATH=src uv run python -m mrrp.features.build_features --feature-config configs/regime_features.yaml

feature-check:
	PYTHONPATH=src uv run python -m mrrp.features.validate_features

models:
	PYTHONPATH=src uv run python -m mrrp.models.build_models --config configs/regime_models.yaml

model-check:
	uv run python -c "from pathlib import Path; p=Path('data/processed/regime_models/manifest.json'); assert p.exists(), p"

stress:
	PYTHONPATH=src uv run python -m mrrp.risk.run_stress

backtest:
	PYTHONPATH=src uv run python -m mrrp.backtest.run_backtest

report:
	PYTHONPATH=src uv run python -m mrrp.reporting.run_report

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
