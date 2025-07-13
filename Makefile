.PHONY: setup clean precommit

# One-step environment setup: create venv, install runtime + dev deps
setup:
	python -m venv .venv
	. .venv/bin/activate && \
	pip install --upgrade pip setuptools wheel && \
	pip install -e .[dev]

# Cleanup artifacts
clean:
	rm -rf .venv build dist *.egg-info

# Run all pre-commit hooks without committing
precommit:
	. .venv/bin/activate && pre-commit run --all-files
