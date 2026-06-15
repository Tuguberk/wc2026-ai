.PHONY: setup update app test lint clean

setup:
	pip install -e ".[dev]"

update:
	python -m wc26.update

app:
	streamlit run app/streamlit_app.py

test:
	pytest tests/ -v --tb=short

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
