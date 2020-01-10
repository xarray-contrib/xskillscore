#!/bin/bash
set -e
set -eo pipefail

echo "Code Styling with (black and flake8)"

source activate xskillscore-dev

echo "[flake8]"
flake8 xskillscore --max-line-length=88 --exclude=__init__.py --ignore=W605,W503,C901,E711

echo "[black]"
black --check --line-length=88 -S xskillscore
