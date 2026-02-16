.PHONY: help install test ci train clean lint

# Default target
help:
	@echo "Available targets:"
	@echo "  install   Install dependencies using uv"
	@echo "  test      Run tests without coverage"
	@echo "  ci        Run tests in CI mode (with XML coverage)"
	@echo "  clean     Clean up temporary files"
	@echo "  tensorboard Launch tensorboard"

install:
	uv sync

test:
	uv run pytest tests/

ci:
	uv run pytest tests/ --maxfail=3 --disable-warnings --tb=short --cov=src/main/python --cov-report=term-missing --cov-report=xml

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".coverage" -exec rm -rf {} +
	rm -f coverage.xml

tensorboard:
	uv run tensorboard --logdir ~/results/marl-swarm-evasion/ray_results