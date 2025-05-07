#!/usr/bin/env bash

conda uninstall -y --force \
    numpy \
    pandas \
    dask \
    cftime \
    bottleneck \
    scipy \
    xarray \
    xhistogram \
    xskillscore

# to limit the runtime of Upstream CI
python -m pip install \
    -i https://pypi.anaconda.org/scientific-python-nightly-wheels/simple \
    --no-deps \
    --pre \
    --upgrade \
    numpy \
    scipy \
    matplotlib \
    pandas
python -m pip install \
    --no-deps \
    --upgrade \
    git+https://github.com/dask/dask \
    git+https://github.com/Unidata/cftime \
    git+https://github.com/pydata/xarray \
    git+https://github.com/pydata/bottleneck \
    git+https://github.com/xgcm/xhistogram
