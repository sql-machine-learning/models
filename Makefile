SHELL := /bin/bash

setup: ## Setup virtual environment for local development
	python3 -m venv venv
	source venv/bin/activate \
	&& $(MAKE) install-requirements

install-requirements:
	pip install --default-timeout=1000 -U -e .

test: ## Run tests
	python3 setup.py test

clean: ## Clean up temporary folders
	rm -rf build dist .eggs *.egg-info .pytest_cache sqlflow/proto

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY: help
.DEFAULT_GOAL := help
