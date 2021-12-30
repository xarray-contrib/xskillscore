from __future__ import annotations

from typing import List, Union

import xarray as xr

XArray = xr.Dataset | xr.DataArray
Dim = Union[str, List[str], None]
# Dim = str | List[str] | None
