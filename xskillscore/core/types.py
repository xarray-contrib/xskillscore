from __future__ import annotations

from typing import List, Optional, Union

import xarray as xr

XArray = Union[xr.Dataset, xr.DataArray]
# XArray = xr.Dataset | xr.DataArray
Dim = Optional[Union[str, List[str]]]
# Dim = str | List[str] | None
