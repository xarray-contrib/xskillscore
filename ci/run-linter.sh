#!/bin/bash
set -e
set -eo pipefail

echo "Code Styling with (isort, black, flake8)"

source activate xskillscore-dev

echo "[isort]"
isort --check-only xskillscore

echo "[black]"
black -S xskillscore

echo "[flake8]"
flake8 xskillscore --exclude=__init__.py

echo "[doc8]"
doc8 *.rst
