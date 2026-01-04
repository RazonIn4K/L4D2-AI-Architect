#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${PROJECT_DIR}/venv"

if [[ -f "${VENV_DIR}/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
else
  echo "Virtual environment not found at: ${VENV_DIR}" >&2
  echo "Create it with:" >&2
  echo "  python3 -m venv venv" >&2
  echo "  source venv/bin/activate" >&2
  echo "  pip install -r requirements.txt" >&2
  return 1 2>/dev/null || exit 1
fi

export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

echo "Activated environment: ${VENV_DIR}"
