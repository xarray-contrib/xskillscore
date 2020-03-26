#!/bin/bash
set -e
set -eo pipefail

echo "Code Styling with (black, flake8)"

source activate xskillscore-dev

echo "[flake8]"
flake8 xskillscore --max-line-length=88 --exclude=__init__.py --ignore=W605,W503,C901,E711

echo "[black]"
black --check --line-length=88 -S xskillscore

#echo "[isort]"
#isort --recursive --check-only xskillscore

#echo "[doc8]"
#doc8 docs/source
#doc8 *.rst
