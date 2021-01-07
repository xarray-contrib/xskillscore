from functools import wraps

import numpy as np
import xarray as xr

from .utils import histogram

__all__ = ["Contingency"]

OBSERVATIONS_NAME = "observations"
FORECASTS_NAME = "forecasts"


def _get_category_bounds(category_edges):
    """Return formatted string of category bounds given list of category edges"""
    return [
        f"[{str(category_edges[i])}, {str(category_edges[i + 1])})"
        for i in range(len(category_edges) - 1)
    ]


def dichotomous_only(method):
    """Decorator for methods that are defined for dichotomous forecasts only"""

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if not self.dichotomous:
            raise AttributeError(
                f"{method.__name__} can only be computed for \
                    dichotomous (2-category) data"
            )
        return method(self, *args, **kwargs)

    return wrapper


def _display_metadata(self):
    """Called when Contingency objects are printed"""
    header = f"<xskillscore.{type(self).__name__}>\n"
    summary = header + "\n".join(str(self.table).split("\n")[1:]) + "\n"
    return summary


class Contingency:
    """Class for contingency based skill scores

    Parameters
    ----------
    observations : xarray.Dataset or xarray.DataArray
        Labeled array(s) over which to apply the function.
    forecasts : xarray.Dataset or xarray.DataArray
        Labeled array(s) over which to apply the function.
    observation_category_edges : array_like
        Bin edges for categorising observations.
        Bins include the left most edge, but not the right.
    forecast_category_edges : array_like
        Bin edges for categorising forecasts.
        Bins include the left most edge, but not the right.
    dim : str, list
        The dimension(s) over which to compute the contingency table

    Returns
    -------
    xskillscore.Contingency

    Examples
    --------
    >>> a = xr.DataArray(
    ...     np.random.normal(size=(3, 3)),
            coords=[('x', np.arange(3)), ('y', np.arange(3))]
    ... ).to_dataset(name='test1')
    >>> b = xr.DataArray(
    ...    np.random.normal(size=(3, 3)),
    ...    coords=[('x', np.arange(3)), ('y', np.arange(3))]
    ... ).to_dataset(name='test1')
    >>> a['test2'] = xr.DataArray(np.random.normal(size=(3,3)),
    ...                           coords=[('x', np.arange(3)), ('y', np.arange(3))])
    >>> b['test2'] = xr.DataArray(np.random.normal(size=(3,3)),
    ...                           coords=[('x', np.arange(3)), ('y', np.arange(3))])
    >>> category_edges_a = np.linspace(-2,2,5)
    >>> category_edges_b = np.linspace(-3,3,5)
    >>> Contingency(a, b, category_edges_a, category_edges_b, dim=['x','y'])
    <xskillscore.Contingency>
    Dimensions:                       (forecasts_category: 4, observations_category: 4)
    Coordinates:
        observations_category_bounds  (observations_category) <U12 '(-2.0, -1.0]'...
        forecasts_category_bounds     (forecasts_category) <U12 '(-3.0, -1.5]' .....
      * observations_category         (observations_category) int64 1 2 3 4
      * forecasts_category            (forecasts_category) int64 1 2 3 4
    Data variables:
        test2                         (observations_category, forecasts_category) int64
        test1                         (observations_category, forecasts_category) int64

    References
    ----------
    http://www.cawcr.gov.au/projects/verification/
    """

    def __init__(
        self,
        observations,
        forecasts,
        observation_category_edges,
        forecast_category_edges,
        dim,
    ):
        self._observations = observations.copy()
        self._forecasts = forecasts.copy()
        self._observation_category_edges = observation_category_edges.copy()
        self._forecast_category_edges = forecast_category_edges.copy()
        self._dichotomous = (
            True
            if (len(observation_category_edges) - 1 == 2)
            & (len(forecast_category_edges) - 1 == 2)
            else False
        )
        self._table = self._get_contingency_table(dim)

    @property
    def observations(self):
        return self._observations

    @property
    def forecasts(self):
        return self._forecasts

    @property
    def observation_category_edges(self):
        return self._observation_category_edges

    @property
    def forecast_category_edges(self):
        return self._forecast_category_edges

    @property
    def dichotomous(self):
        return self._dichotomous

    @property
    def table(self):
        return self._table

    def _get_contingency_table(self, dim):
        """Build the contingency table

        Parameters
        ----------
        dim : str, list
            The dimension(s) over which to compute the contingency table

        Returns
        -------
        xarray.Dataset or xarray.DataArray
        """

        table = histogram(
            self.observations,
            self.forecasts,
            bins=[self.observation_category_edges, self.forecast_category_edges],
            bin_names=[OBSERVATIONS_NAME, FORECASTS_NAME],
            dim=dim,
            bin_dim_suffix="_bin",
        )

        # Add some coordinates to simplify interpretation/post-processing
        table = table.assign_coords(
            {
                OBSERVATIONS_NAME
                + "_bin": _get_category_bounds(self.observation_category_edges)
            }
        ).rename({OBSERVATIONS_NAME + "_bin": OBSERVATIONS_NAME + "_category_bounds"})
        table = table.assign_coords(
            {
                FORECASTS_NAME
                + "_bin": _get_category_bounds(self.forecast_category_edges)
            }
        ).rename({FORECASTS_NAME + "_bin": FORECASTS_NAME + "_category_bounds"})
        table = table.assign_coords(
            {
                OBSERVATIONS_NAME
                + "_category": (
                    OBSERVATIONS_NAME + "_category_bounds",
                    range(1, len(self.observation_category_edges)),
                ),
                FORECASTS_NAME
                + "_category": (
                    FORECASTS_NAME + "_category_bounds",
                    range(1, len(self.forecast_category_edges)),
                ),
            }
        )
        table = table.swap_dims(
            {
                OBSERVATIONS_NAME + "_category_bounds": OBSERVATIONS_NAME + "_category",
                FORECASTS_NAME + "_category_bounds": FORECASTS_NAME + "_category",
            }
        )

        return table

    def _sum_categories(self, categories):
        """Returns sums of specified categories in contingency table

        Parameters
        ----------
        category : str, optional
            Contingency table categories to sum.
            Options are 'total', 'observations' and 'forecasts'

        Returns
        -------
        Sum of all counts in specified categories

        """

        if categories == "total":
            N = self.table.sum(
                dim=(OBSERVATIONS_NAME + "_category", FORECASTS_NAME + "_category"),
                skipna=True,
            )
        elif categories == "observations":
            N = self.table.sum(dim=FORECASTS_NAME + "_category", skipna=True).rename(
                {OBSERVATIONS_NAME + "_category": "category"}
            )
        elif categories == "forecasts":
            N = self.table.sum(dim=OBSERVATIONS_NAME + "_category", skipna=True).rename(
                {FORECASTS_NAME + "_category": "category"}
            )
        else:
            raise ValueError(
                f"'{categories}' is not a recognised category. \
                    Pick one of ['total', 'observations', 'forecasts']"
            )

        return N

    def __repr__(self):
        return _display_metadata(self)

    @dichotomous_only
    def hits(self, yes_category=2):
        """Returns the number of hits (true positives) for dichotomous
        contingency data.

        Parameters
        ----------
        yes_category : value, optional
            The category coordinate value of the category corresponding to 'yes'

        Returns
        -------
        xarray.Dataset or xarray.DataArray
            An array containing the number of hits

        References
        ----------
        https://www.cawcr.gov.au/projects/verification/#Contingency_table
        """

        return self.table.sel(
            {
                OBSERVATIONS_NAME + "_category": yes_category,
                FORECASTS_NAME + "_category": yes_category,
            },
            drop=True,
        )

    @dichotomous_only
    def misses(self, yes_category=2):
        """Returns the number of misses (false negatives) for dichotomous
        contingency data.

        Parameters
        ----------
        yes_category : value, optional
            The category coordinate value of the category corresponding to 'yes'

        Returns
        -------
        xarray.Dataset or xarray.DataArray
            An array containing the number of misses

        References
        ----------
        https://www.cawcr.gov.au/projects/verification/#Contingency_table
        """
        no_category = abs(yes_category - 2) + 1

        return self.table.sel(
            {
                OBSERVATIONS_NAME + "_category": yes_category,
                FORECASTS_NAME + "_category": no_category,
            },
            drop=True,
        )

    @dichotomous_only
    def false_alarms(self, yes_category=2):
        """Returns the number of false alarms (false positives) for dichotomous
        contingency data.

        Parameters
        ----------
        yes_category : value, optional
            The category coordinate value of the category corresponding to 'yes'

        Returns
        -------
        xarray.Dataset or xarray.DataArray
            An array containing the number of false alarms

        References
        ----------
        https://www.cawcr.gov.au/projects/verification/#Contingency_table
        """
        no_category = abs(yes_category - 2) + 1

        return self.table.sel(
            {
                OBSERVATIONS_NAME + "_category": no_category,
                FORECASTS_NAME + "_category": yes_category,
            },
            drop=True,
        )

    @dichotomous_only
    def correct_negatives(self, yes_category=2):
        """Returns the number of correct negatives (true negatives) for dichotomous
        contingency data.

        Parameters
        ----------
        yes_category : value, optional
            The category coordinate value of the category corresponding to 'yes'

        Returns
        -------
        xarray.Dataset or xarray.DataArray
            An array containing the number of correct negatives

        References
        ----------
        https://www.cawcr.gov.au/projects/verification/#Contingency_table
        """
        no_category = abs(yes_category - 2) + 1

        return self.table.sel(
            {
                OBSERVATIONS_NAME + "_category": no_category,
                FORECASTS_NAME + "_category": no_category,
            },
            drop=True,
        )

    @dichotomous_only
    def bias_score(self, yes_category=2):
        """Returns the bias score(s) for dichotomous contingency data

        .. math::

            BS = \\frac{\\mathrm{hits} + \\mathrm{false~alarms}}
                 {\\mathrm{hits} + \\mathrm{misses}}

        Parameters
        ----------
        yes_category : value, optional
            The category coordinate value of the category corresponding to 'yes'

        Returns
        -------
        xarray.Dataset or xarray.DataArray
            An array containing the bias score(s)

        References
        ----------
        https://www.cawcr.gov.au/projects/verification/#Contingency_table
        """

        return (self.hits(yes_category) + self.false_alarms(yes_category)) / (
            self.hits(yes_category) + self.misses(yes_category)
        )

    @dichotomous_only
    def hit_rate(self, yes_category=2):
        """Returns the hit rate(s) (probability of detection) for
        dichotomous contingency data.

        .. math::
            HR = \\frac{hits}{hits + misses}

        Parameters
        ----------
        yes_category : value, optional
            The category coordinate value of the
            category corresponding to 'yes' (1 or 2)

        Returns
        -------
        xarray.Dataset or xarray.DataArray
            An array containing the hit rate(s)

        References
        ----------
        https://www.cawcr.gov.au/projects/verification/#Contingency_table
        """

        return self.hits(yes_category) / (
            self.hits(yes_category) + self.misses(yes_category)
        )

    @dichotomous_only
    def false_alarm_ratio(self, yes_category=2):
        """Returns the false alarm ratio(s) for dichotomous contingency data.

        .. math::
            FAR = \\frac{\\mathrm{false~alarms}}{hits + \\mathrm{false~alarms}}

        Parameters
        ----------
        yes_category : value, optional
            The category coordinate value of the category corresponding to 'yes'

        Returns
        -------
        xarray.Dataset or xarray.DataArray
            An array containing the false alarm ratio(s)

        References
        ----------
        https://www.cawcr.gov.au/projects/verification/#Contingency_table
        """

        return self.false_alarms(yes_category) / (
            self.hits(yes_category) + self.false_alarms(yes_category)
        )

    @dichotomous_only
    def false_alarm_rate(self, yes_category=2):
        """Returns the false alarm rate(s) (probability of false detection)
        for dichotomous contingency data.

        .. math::
            FA = \\frac{\\mathrm{false~alarms}}
                 {\\mathrm{correct~negatives} + \\mathrm{false~alarms}}

        Parameters
        ----------
        yes_category : value, optional
            The category coordinate value of the category corresponding to 'yes'

        Returns
        -------
        xarray.Dataset or xarray.DataArray
            An array containing the false alarm rate(s)

        References
        ----------
        https://www.cawcr.gov.au/projects/verification/#Contingency_table
        """

        return self.false_alarms(yes_category) / (
            self.correct_negatives(yes_category) + self.false_alarms(yes_category)
        )

    @dichotomous_only
    def success_ratio(self, yes_category=2):
        """Returns the success ratio(s) for dichotomous contingency data.

        .. math::
            SR = \\frac{hits}{hits + \\mathrm{false~alarms}}

        Parameters
        ----------
        yes_category : value, optional
            The category coordinate value of the category corresponding to 'yes'

        Returns
        -------
        xarray.Dataset or xarray.DataArray
            An array containing the success ratio(s)

        References
        ----------
        https://www.cawcr.gov.au/projects/verification/#Contingency_table
        """

        return self.hits(yes_category) / (
            self.hits(yes_category) + self.false_alarms(yes_category)
        )

    @dichotomous_only
    def threat_score(self, yes_category=2):
        """Returns the threat score(s) for dichotomous contingency data.

        .. math::
            TS = \\frac{hits}{hits + misses + \\mathrm{false~alarms}}

        Parameters
        ----------
        yes_category : value, optional
            The category coordinate value of the category corresponding to 'yes'

        Returns
        -------
        xarray.Dataset or xarray.DataArray
            An array containing the threat score(s)

        References
        ----------
        https://www.cawcr.gov.au/projects/verification/#Contingency_table
        """

        return self.hits(yes_category) / (
            self.hits(yes_category)
            + self.misses(yes_category)
            + self.false_alarms(yes_category)
        )

    @dichotomous_only
    def equit_threat_score(self, yes_category=2):
        """Returns the equitable threat score(s) for dichotomous contingency data.

        .. math::
            ETS = \\frac{hits - hits_{random}}
                  {hits + misses + \\mathrm{false~alarms} - hits_{random}}

        .. math::
            hits_{random} = \\frac{(hits + misses
                            (hits + \\mathrm{false~alarms})}{total}

        Parameters
        ----------
        yes_category : value, optional
            The category coordinate value of the category corresponding to 'yes'

        Returns
        -------
        xarray.Dataset or xarray.DataArray
            An array containing the equitable threat score(s)

        References
        ----------
        https://www.cawcr.gov.au/projects/verification/#Contingency_table
        """

        hits_random = (
            (self.hits(yes_category) + self.misses(yes_category))
            * (self.hits(yes_category) + self.false_alarms(yes_category))
        ) / self._sum_categories("total")

        return (self.hits(yes_category) - hits_random) / (
            self.hits(yes_category)
            + self.misses(yes_category)
            + self.false_alarms(yes_category)
            - hits_random
        )

    @dichotomous_only
    def odds_ratio(self, yes_category=2):
        """Returns the odds ratio(s) for dichotomous contingency data

        .. math::
            OR = \\frac{hits * \\mathrm{correct~negatives}}
                       {misses * \\mathrm{false~alarms}}

        Parameters
        ----------
        yes_category : value, optional
            The category coordinate value of the category corresponding to 'yes'

        Returns
        -------
        xarray.Dataset or xarray.DataArray
            An array containing the equitable odds ratio(s)

        References
        ----------
        https://www.cawcr.gov.au/projects/verification/#Contingency_table
        """

        return (self.hits(yes_category) * self.correct_negatives(yes_category)) / (
            self.misses(yes_category) * self.false_alarms(yes_category)
        )

    @dichotomous_only
    def odds_ratio_skill_score(self, yes_category=2):
        """Returns the odds ratio skill score(s) for dichotomous contingency data

        .. math::
            ORSS = \\frac{hits * \\mathrm{correct~negatives}
                      - misses * \\mathrm{false~alarms}}
            {hits * \\mathrm{correct~negatives} + misses * \\mathrm{false~alarms}}

        Parameters
        ----------
        yes_category : value, optional
            The category coordinate value of the category corresponding to 'yes'

        Returns
        -------
        xarray.Dataset or xarray.DataArray
            An array containing the equitable odds ratio skill score(s)

        References
        ----------
        https://www.cawcr.gov.au/projects/verification/#Contingency_table
        """

        return (
            self.hits(yes_category) * self.correct_negatives(yes_category)
            - self.misses(yes_category) * self.false_alarms(yes_category)
        ) / (
            self.hits(yes_category) * self.correct_negatives(yes_category)
            + self.misses(yes_category) * self.false_alarms(yes_category)
        )

    def accuracy(self):
        """Returns the accuracy score(s) for a contingency table with K categories

        .. math::
            A = \\frac{1}{N}\\sum_{i=1}^{K} n(F_i, O_i)

        Returns
        -------
        xarray.Dataset or xarray.DataArray
            An array containing the accuracy score(s)

        References
        ----------
        https://www.cawcr.gov.au/projects/verification/#Contingency_table
        """

        corr = self.table.where(
            self.table[OBSERVATIONS_NAME + "_category"]
            == self.table[FORECASTS_NAME + "_category"]
        ).sum(
            dim=(OBSERVATIONS_NAME + "_category", FORECASTS_NAME + "_category"),
            skipna=True,
        )
        N = self._sum_categories("total")

        return corr / N

    def heidke_score(self):
        """Returns the Heidke skill score(s) for a contingency table with K categories

        .. math::
            HSS = \\frac{\\frac{1}{N}\\sum_{i=1}^{K}n(F_i, O_i) -
                  \\frac{1}{N^2}\\sum_{i=1}^{K}N(F_i)N(O_i)}
                  {1 - \\frac{1}{N^2}\\sum_{i=1}^{K}N(F_i)N(O_i)}

        Returns
        -------
        xarray.Dataset or xarray.DataArray
            An array containing the Heidke score(s)

        References
        ----------
        https://www.cawcr.gov.au/projects/verification/#Contingency_table
        """

        numer_1 = (
            self.table.where(
                self.table[OBSERVATIONS_NAME + "_category"]
                == self.table[FORECASTS_NAME + "_category"]
            ).sum(
                dim=(OBSERVATIONS_NAME + "_category", FORECASTS_NAME + "_category"),
                skipna=True,
            )
            / self._sum_categories("total")
        )
        numer_2 = (
            self._sum_categories("observations") * self._sum_categories("forecasts")
        ).sum(dim="category", skipna=True) / self._sum_categories("total") ** 2
        denom = 1 - numer_2

        return (numer_1 - numer_2) / denom

    def peirce_score(self):
        """Returns the Peirce skill score(s) (Hanssen and Kuipers discriminantor true
        skill statistic) for a contingency table with K categories.

        .. math::
            PS = \\frac{\\frac{1}{N}\\sum_{i=1}^{K}n(F_i, O_i) -
                 \\frac{1}{N^2}\\sum_{i=1}^{K}N(F_i)N(O_i)}{1 -
                 \\frac{1}{N^2}\\sum_{i=1}^{K}N(O_i)^2}

        Returns
        -------
        xarray.Dataset or xarray.DataArray
            An array containing the Peirce score(s)

        References
        ----------
        https://www.cawcr.gov.au/projects/verification/#Contingency_table
        """

        numer_1 = (
            self.table.where(
                self.table[OBSERVATIONS_NAME + "_category"]
                == self.table[FORECASTS_NAME + "_category"]
            ).sum(
                dim=(OBSERVATIONS_NAME + "_category", FORECASTS_NAME + "_category"),
                skipna=True,
            )
            / self._sum_categories("total")
        )
        numer_2 = (
            self._sum_categories("observations") * self._sum_categories("forecasts")
        ).sum(dim="category", skipna=True) / self._sum_categories("total") ** 2
        denom = 1 - (self._sum_categories("observations") ** 2).sum(
            dim="category", skipna=True
        ) / (self._sum_categories("total") ** 2)

        return (numer_1 - numer_2) / denom

    def gerrity_score(self):
        """Returns Gerrity equitable score for a contingency table with K categories.

        .. math::
                GS = \\frac{1}{N}\\sum_{i=1}^{K}\\sum_{j=1}^{K}n(F_i, O_j)s_{ij}

        .. math::
                s_{ii} = \\frac{1}{K-1}(\\sum_{r=1}^{i-1}a_r^{-1} +
                \\sum_{r=i}^{K-1}a_r)

        .. math::
                s_{ij} = \\frac{1}{K-1}(\\sum_{r=1}^{i-1}a_r^{-1} - (j - i) +
                \\sum_{r=j}^{K-1}a_r); 1 \\leq i < j \\leq K

        .. math::
            s_{ji} = s_{ij}

        .. math::
            a_i = \\frac{(1 - \\sum_{r=1}^{i}p_r)}{\\sum_{r=1}^{i}p_r}

        .. math::
            p_i = \\frac{N(O_i)}{N}

        Returns
        -------
        xarray.Dataset or xarray.DataArray
            An array containing the Gerrity scores

        References
        ----------
        https://www.cawcr.gov.au/projects/verification/#Contingency_table
        """
        # TODO: Currently computes the Gerrity scoring matrix using nested for-loops.
        # Is it possible to remove these?

        def _gerrity_s(table):
            """Returns Gerrity scoring matrix, s"""
            p_o = (table.sum(axis=-1).T / table.sum(axis=(-2, -1)).T).T
            p_sum = np.cumsum(p_o, axis=-1)
            a = (1.0 - p_sum) / p_sum
            k = a.shape[-1]
            s = np.zeros(table.shape, dtype=float)
            for (i, j) in np.ndindex(*s.shape[-2:]):
                if i == j:
                    s[..., i, j] = (
                        1.0
                        / (k - 1.0)
                        * (
                            np.sum(1.0 / a[..., 0:j], axis=-1)
                            + np.sum(a[..., j : k - 1], axis=-1)
                        )
                    )
                elif i < j:
                    s[..., i, j] = (
                        1.0
                        / (k - 1.0)
                        * (
                            np.sum(1.0 / a[..., 0:i], axis=-1)
                            - (j - i)
                            + np.sum(a[..., j : k - 1], axis=-1)
                        )
                    )
                else:
                    s[..., i, j] = s[..., j, i]
            return s

        s = xr.apply_ufunc(
            _gerrity_s,
            self.table,
            input_core_dims=[
                [OBSERVATIONS_NAME + "_category", FORECASTS_NAME + "_category"]
            ],
            output_core_dims=[
                [OBSERVATIONS_NAME + "_category", FORECASTS_NAME + "_category"]
            ],
            dask="allowed",
        )

        return (self.table * s).sum(
            dim=(OBSERVATIONS_NAME + "_category", FORECASTS_NAME + "_category"),
            skipna=True,
        ) / self._sum_categories("total")


def roc(
    observations,
    forecasts,
    bin_edges,
    dim=None,
    drop_intermediate=True,
    return_results="area",
):
    """Computes the relative operating characteristic of an event for a range of bin edges.

    Parameters
    ----------
    observations : xarray.Dataset or xarray.DataArray
        Labeled array(s) over which to apply the function.
        If bin_edges=='continuous', observations are binary.
    forecasts : xarray.Dataset or xarray.DataArray
        Labeled array(s) over which to apply the function.
        If bin_edges=='continuous', forecasts are probabilities.
    bin_edges : array_like, str
        Bin edges for categorising observations.
        Bins include the left most edge, but not the right.
         or
        'continuous': to match sklearn.metrics.roc_curve(f_boolean, o_prob, drop_intermediate=False)
    dim : str, list
        The dimension(s) over which to compute the contingency table
    drop_intermediate : bool, default=True
        Whether to drop some suboptimal thresholds which would not appear
        on a plotted ROC curve. This is useful in order to create lighter
        ROC curves.
    return_results: str, default='area'
        Specify how return is structed:
        - 'area': return only the area under the curve of ROC
        - 'all_as_tuple': return hit and false alarm rate at each bin and area under the curve of ROC as tuple
        - 'all_as_metric_dim': return hit and false alarm rate at each bin and area under the curve of ROC
            concatinated into new `metric` dimension

    Returns
    -------
    xarray.Dataset or xarray.DataArray : reduced by dimensions ``dim`` and specified by ``return_results``

    Examples
    --------
    >>> f = xr.DataArray(
    ...     np.random.normal(size=(1000)),
            coords=[('time', np.arange(1000))]
    ... )
    >>> o = xr.DataArray(
    ...    np.random.normal(size=(1000)),
    ...    coords=[('time', np.arange(1000))]
    ... )
    >>> category_edges = np.linspace(-2,2,5)
    >>> roc(a, b, category_edges, dim=['time'])
    <xarray.DataArray 'histogram_observations_forecasts' ()>
    array(0.46812223)

    See also
    --------
    xskillscore.Contingency

    References
    ----------
    http://www.cawcr.gov.au/projects/verification/
    """

    if dim is None:
        dim = list(forecasts.dims)
    if isinstance(dim, str):
        dim = [dim]

    continuous = False
    if isinstance(bin_edges, str):
        if bin_edges == "continuous":
            continuous = True
            # check that o binary
            if isinstance(observations, xr.Dataset):
                o_check = observations.to_array()
            else:
                o_check = observations
            if str(o_check.dtype) != "bool":
                if not ((o_check == 0) | (o_check == 1)).all():
                    raise ValueError(
                        'Input "observations" must represent logical (True/False) outcomes',
                        o_check,
                    )

            # works only for 1var
            if isinstance(forecasts, xr.Dataset):
                v = list(forecasts.data_vars)[0]
                f_bin = forecasts[v]
            else:
                f_bin = forecasts
            f_bin = f_bin.stack(ndim=forecasts.dims)
            f_bin = f_bin.sortby(-f_bin)
            bin_edges = np.append(f_bin[0] + 1, f_bin)
            bin_edges = np.unique(bin_edges)[::-1]
            # print(bin_edges)

    # loop over each bin_edge and get hit rate and false alarm rate from contingency
    hr, far = [], []
    for i in bin_edges:
        dichotomous_category_edges = np.array(
            [-np.inf, i, np.inf]
        )  # "dichotomous" means two-category
        dichotomous_contingency = Contingency(
            observations,
            forecasts,
            dichotomous_category_edges,
            dichotomous_category_edges,
            dim=dim,
        )
        far.append(dichotomous_contingency.false_alarm_rate())
        hr.append(dichotomous_contingency.hit_rate())
    hr = xr.concat(hr, "probability_bin")
    far = xr.concat(far, "probability_bin")

    easyfillna = True
    if easyfillna:
        far = far.fillna(1.0)
        hr = hr.fillna(0.0)

    # pad (0,0) and (1,1)
    far_pad = xr.concat(
        [
            xr.ones_like(far.isel(probability_bin=0, drop=True)),
            far,
            xr.zeros_like(far.isel(probability_bin=-1, drop=True)),
        ],
        "probability_bin",
    )  # .sortby(far)
    hr_pad = xr.concat(
        [
            xr.ones_like(hr.isel(probability_bin=0, drop=True)),
            hr,
            xr.zeros_like(hr.isel(probability_bin=-1, drop=True)),
        ],
        "probability_bin",
    )  # .sortby(hr)
    # far=far.bfill('probability_bin').ffill('probability_bin')
    # hr=hr.ffill('probability_bin').bfill('probability_bin')

    # https://github.com/scikit-learn/scikit-learn/blob/42aff4e2edd8e8887478f6ff1628f27de97be6a3/sklearn/metrics/_ranking.py#L916
    # Attempt to drop thresholds corresponding to points in between and
    # collinear with other points. These are always suboptimal and do not
    # appear on a plotted ROC curve (and thus do not affect the AUC).
    # Here np.diff(_, 2) is used as a "second derivative" to tell if there
    # is a corner at the point. Both fps and tps must be tested to handle
    # thresholds with multiple data points (which are combined in
    # _binary_clf_curve). This keeps all cases where the point should be kept,
    # but does not drop more complicated cases like fps = [1, 3, 7],
    # tps = [1, 2, 4]; there is no harm in keeping too many thresholds.
    if drop_intermediate and far.probability_bin.size > 2:

        def drop_interm(far, hr):
            if isinstance(far, xr.Dataset):
                if len(far.data_vars) == 1:
                    v = list(far.data_vars)[0]
                    hr_check = hr[v]
                    far_check = far[v]
                else:
                    raise ValueError(
                        "drop_intermediate=True only works for one variable xr.Dataset or xr.DataArray."
                    )
            else:
                hr_check = hr
                far_check = far
            optimal_idxs = np.where(
                np.r_[
                    True,
                    np.logical_or(
                        far_check.diff("probability_bin", 2),
                        hr_check.diff("probability_bin", 2),
                    ),
                    True,
                ]
            )[0]
            hr = hr.isel(probability_bin=optimal_idxs)
            far = far.isel(probability_bin=optimal_idxs)
            return far, hr

        far, hr = drop_interm(far, hr)
        far_pad, hr_pad = drop_interm(far_pad, hr_pad)

    def auc(hr, far, dim="probability_bin"):
        """Get area under the curve with trapez method."""
        area = xr.apply_ufunc(
            np.trapz, -hr, far, input_core_dims=[[dim], [dim]], dask="allowed"
        )
        area = np.clip(area, 0, 1)  # allow only values between 0 and 1
        return area

    area = auc(hr_pad, far_pad)
    if continuous:
        area = 1 - area  # dirty fix

    # mask always nan
    def _keep_masked(new, ori, dim):
        """Keep mask from `ori` deprived of dimensions from `dim` in input `new`."""
        isel_dim = {d: 0 for d in forecasts.dims if d in dim}
        mask = ori.isel(isel_dim, drop=True)
        new_masked = new.where(mask.notnull())
        return new_masked

    far = _keep_masked(far, forecasts, dim=dim)
    hr = _keep_masked(hr, forecasts, dim=dim)
    area = _keep_masked(area, forecasts, dim=dim)

    if return_results == "area":
        return area
    elif return_results == "all_as_metric_dim":
        results = xr.concat([far, hr, area], "metric", coords="minimal")
        results["metric"] = ["false alarm rate", "hit rate", "area under curve"]
        return results
    elif return_results == "all_as_tuple":
        return far, hr, area
    else:
        raise NotImplementedError(
            f"expect `return_results` from [all_as_tuple, area, all_as_metric_dim], found {return_results}"
        )
