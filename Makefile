# L4D2-AI-Architect Makefile

VENV := venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

.PHONY: help setup install scrape-github scrape-wiki process-data start-ui clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: ## Create venv and install dependencies
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "Setup complete. Activate with: source $(VENV)/bin/activate"

scrape-github: ## Scrape GitHub plugins
	$(PYTHON) scripts/scrapers/scrape_github_plugins.py --max-repos 500

scrape-wiki: ## Scrape Valve Developer Wiki
	$(PYTHON) scripts/scrapers/scrape_valve_wiki.py

process-data: ## Prepare training dataset
	$(PYTHON) scripts/training/prepare_dataset.py --input data/raw --output data/processed

train-unsloth: ## Run Unsloth training (Linux/GPU only)
	./scripts/launchers/run_training.sh

bot-agent: ## Run RL Bot Agent
	./scripts/launchers/run_ai_bot.sh

ai-director: ## Run AI Director
	./scripts/launchers/run_ai_director.sh

web-ui: ## Start the Web UI
	./scripts/launchers/start_web_ui.sh

test: ## Run tests
	$(PYTHON) -m pytest tests/

clean: ## Clean up temporary files
	rm -rf __pycache__ .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
