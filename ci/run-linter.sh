#!/bin/bash
set -e
set -eo pipefail

echo "Code Styling with (black, flake8, isort)"

source activate xskillscore-dev

echo "[flake8]"
flake8 xskillscore --max-line-length=88 --exclude=__init__.py --ignore=E203,E266,E501,W503,F401,W605,E402,C901

echo "[black]"
black --check --line-length=88 -S xskillscore

echo "[isort]"
isort --recursive --check-only xskillscore

echo "[doc8]"
doc8 *.rst
