from __future__ import annotations

from typing import List, Union

import xarray as xr

XArray = Union[xr.Dataset, xr.DataArray]
# XArray = xr.Dataset | xr.DataArray # raises error during build: TypeError: unsupported operand type(s) for |: 'ABCMeta' and 'type'
Dim = Union[str, List[str], None]
# Dim = str | List[str] | None
