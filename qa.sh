#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
[ -x .venv/bin/python ] || { echo ".venv missing – see README" >&2; exit 1; }

.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/python -m pytest -q
