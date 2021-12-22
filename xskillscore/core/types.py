from typing import List, Optional, Union

import xarray as xr

XArray = Union[xr.Dataset, xr.DataArray]
Dim = Optional[Union[str, List[str]]]
