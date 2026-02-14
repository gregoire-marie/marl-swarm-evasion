.PHONY: help install test ci train clean lint

# Default target
help:
	@echo "Available targets:"
	@echo "  install   Install dependencies using uv"
	@echo "  test      Run tests with coverage"
	@echo "  ci        Run tests in CI mode (with XML coverage)"
	@echo "  train     Run training script"
	@echo "  clean     Clean up temporary files"

install:
	uv sync

test:
	uv run pytest tests/ --cov=src/main/python --cov-report=term-missing

ci:
	uv run pytest tests/ --maxfail=3 --disable-warnings --tb=short --cov=src/main/python --cov-report=term-missing --cov-report=xml

train:
	uv run python app/train.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".coverage" -exec rm -rf {} +
	rm -f coverage.xml
