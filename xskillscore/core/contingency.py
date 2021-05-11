from functools import wraps

import numpy as np
import xarray as xr

from .utils import histogram

__all__ = ["Contingency"]

OBSERVATIONS_NAME = "observations"
FORECASTS_NAME = "forecasts"


def _get_category_bounds(category_edges):
    """Return formatted string of category bounds given list of category edges"""
    bounds = [
        f"[{str(category_edges[i])}, {str(category_edges[i + 1])})"
        for i in range(len(category_edges) - 2)
    ]
    # Last category is right edge inclusive
    bounds.append(f"[{str(category_edges[-2])}, {str(category_edges[-1])}]")
    return bounds


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
        Bin edges for categorising observations. Similar to np.histogram, \
        all but the last (righthand-most) bin include the left edge and \
        exclude the right edge. The last bin includes both edges.
    forecast_category_edges : array_like
        Bin edges for categorising forecasts. Similar to np.histogram, \
        all but the last (righthand-most) bin include the left edge and \
        exclude the right edge. The last bin includes both edges.
    dim : str, list
        The dimension(s) over which to compute the contingency table

    Returns
    -------
    xskillscore.Contingency

    Examples
    --------
    >>> da = xr.DataArray(np.random.normal(size=(3, 3)),
    ...                   coords=[("x", np.arange(3)), ("y", np.arange(3))])
    >>> o = xr.Dataset({"var1": da, "var2": da})
    >>> f = o * 1.1
    >>> o_category_edges = np.linspace(-2, 2, 5)
    >>> f_category_edges = np.linspace(-3, 3, 5)
    >>> xs.Contingency(o, f,
    ...                o_category_edges, f_category_edges,
    ...                dim=['x', 'y']) # doctest: +SKIP
    <xskillscore.Contingency>
    Dimensions:                       (forecasts_category: 4, observations_category: 4)
    Coordinates:
        observations_category_bounds  (observations_category) <U12 '[-2.0, -1.0)'...
        forecasts_category_bounds     (forecasts_category) <U12 '[-3.0, -1.5)' .....
      * observations_category         (observations_category) int64 1 2 3 4
      * forecasts_category            (forecasts_category) int64 1 2 3 4
    Data variables:
        var1                           (observations_category, forecasts_category) int64
        var2                           (observations_category, forecasts_category) int64


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

        See Also
        --------
        sklearn.metrics.recall_score

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

        See Also
        --------
        sklearn.metrics.precision_score

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

        See Also
        --------
        sklearn.metrics.accuracy_score

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

        See Also
        --------
        sklearn.metrics.cohen_kappa_score

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
