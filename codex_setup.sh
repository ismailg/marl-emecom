#!/usr/bin/env bash
set -euo pipefail

# Runtime pin
if command -v python3.9 >/dev/null 2>&1; then
  PY=python3.9
else
  PY=python3
fi

$PY -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel

# Project install (for marl-emecom fork)
if [ -f requirements.txt ]; then
  pip install -r requirements.txt
fi
pip install -e .

# Lock exact environment for reproducibility
pip freeze > requirements_locked.txt
