.PHONY: setup etl audit train test lint format app api docker clean

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -U pip && \
	pip install -r requirements-dev.txt && pip install -e .

etl:
	python -m customer_churn.etl

audit:
	python -m customer_churn.audit

train: etl
	python -m customer_churn.train

test:
	pytest --cov=customer_churn --cov-report=term-missing

lint:
	ruff check src tests app && black --check src tests app

format:
	black src tests app && ruff check --fix src tests app

app:
	streamlit run app/streamlit_app.py

api:
	uvicorn app.api.main:app --reload

docker:
	docker compose up --build

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache **/__pycache__ data/processed/*.duckdb
