#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"

if [[ -f "./venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "./venv/bin/activate"
fi

mkdir -p data/raw data/processed data/training_logs

echo "Running data collection..."
python3 scripts/scrapers/scrape_github_plugins.py --max-repos "${MAX_REPOS:-500}"
python3 scripts/scrapers/scrape_valve_wiki.py --max-pages "${MAX_PAGES:-200}"

echo "Preparing dataset..."
python3 scripts/training/prepare_dataset.py
