.PHONY: setup data dashboard test lint format check clean

setup:
	uv sync

data:
	uv run python scripts/download_data.py --config configs/default_universe.yaml --out data/processed/adjusted_close.parquet

dashboard:
	PYTHONPATH=src uv run streamlit run app/streamlit_app.py

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
