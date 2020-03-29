#!/bin/bash
set -e
set -eo pipefail

echo "Code Styling with (black, flake8, isort)"

source activate xskillscore-dev

echo "[flake8]"
flake8 xskillscore --max-line-length=88 --exclude=__init__.py --ignore=C901,E203,E266,E402,E501,E711,F401,W503,W605

echo "[black]"
black --check --line-length=88 -S xskillscore

echo "[isort]"
isort --recursive --check-only xskillscore

echo "[doc8]"
doc8 *.rst
