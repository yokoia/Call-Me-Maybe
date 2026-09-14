PYTHON := uv run python

.PHONY: install run debug clean lint lint-strict test

install:
	uv sync

run:
	$(PYTHON) -m src

debug:
	$(PYTHON) -m pdb -m src

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache

lint:
	uv run flake8 src
	uv run mypy src \
		--warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs

lint-strict:
	uv run flake8 .
	uv run mypy . --strict

test:
	uv run pytest -q
