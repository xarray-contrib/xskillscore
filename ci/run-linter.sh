#!/bin/bash
set -e
set -eo pipefail

echo "Code Styling with (black, flake8, isort)"

source activate xskillscore-dev

echo "[flake8]"
flake8 xskillscore

echo "[black]"
black --check -S xskillscore

echo "[isort]"
isort --recursive --check-only xskillscore

echo "[doc8]"
doc8 *.rst
