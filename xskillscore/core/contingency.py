from functools import wraps

import numpy as np
import xarray as xr

from xhistogram.xarray import histogram as xhist

__all__ = ['histogram', 'Contingency']

OBSERVATIONS_NAME = 'observations'
FORECASTS_NAME = 'forecasts'


def histogram(*args, bins=None, bin_names=None, **kwargs):
    """
        Wrapper on xhistogram to deal with Datasets appropriately
    """
    if isinstance(args[0], xr.core.dataset.Dataset):
        # Get list of variables that are shared across all Datasets
        overlapping_vars = set.intersection(*map(set, [arg.data_vars for arg in args]))
        if overlapping_vars:
            # If bin_names not provided, use default ----
            if bin_names is None:
                bin_names = ['ds_' + str(i + 1) for i in range(len(args))]
            return xr.merge(
                [
                    xhist(
                        *(arg[var].rename(bin_names[i]) for i, arg in enumerate(args)),
                        bins=bins,
                        **kwargs,
                    ).rename(var)
                    for var in overlapping_vars
                ]
            )
        else:
            raise ValueError('No common variables exist across input Datasets')
    else:
        if bin_names:
            args = (arg.rename(bin_names[i]) for i, arg in enumerate(args))
        return xhist(*args, bins=bins, **kwargs)


def get_category_bounds(category_edges):
    """
        Return formatted string of cateogry bounds given list of category edges
    """
    return [
        '(' + str(category_edges[i]) + ', ' + str(category_edges[i + 1]) + ']'
        for i in range(len(category_edges) - 1)
    ]


def dichotomous_only(method):
    """
        Decorator for methods that are defined for dichonomous forecasts only
    """

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if not self.dichotomous:
            raise AttributeError(
                f'{method.__name__} can only be computed for dichotomous (2-category) data'
            )
        return method(self, *args, **kwargs)

    return wrapper


def _display_metadata(self):
    """
        Called when Contingency objects are printed
    """
    header = f'<xskillscore.{type(self).__name__}>\n'
    summary = header + '\n'.join(str(self.table).split('\n')[1:]) + '\n'
    return summary


class Contingency:
    """
        Class for contingency based skill scores

        Parameters
        ----------
        observations : xarray.Dataset or xarray.DataArray
            Labeled array(s) over which to apply the function.
        forecasts : xarray.Dataset or xarray.DataArray
            Labeled array(s) over which to apply the function.
        observation_category_edges : array_like
            Bin edges for categorising observations
        forecast_category_edges : array_like
            Bin edges for categorising forecasts
        dim : str, list
            The dimension(s) over which to compute the contingency table

        Returns
        -------
        xskillscore.Contingency

        Examples
        --------
        >>> a = xr.DataArray(np.random.normal(size=(3,3)),
        ...                  coords=[('x', np.arange(3)), ('y', np.arange(3))]).to_dataset(name='test1')
        >>> b = xr.DataArray(np.random.normal(size=(3,3)),
        ...                  coords=[('x', np.arange(3)), ('y', np.arange(3))]).to_dataset(name='test1')
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
            test2                         (observations_category, forecasts_category) int64 ...
            test1                         (observations_category, forecasts_category) int64 ...

        Notes
        -----
        See http://www.cawcr.gov.au/projects/verification/
    """

    def __init__(
        self,
        observations,
        forecasts,
        observation_category_edges,
        forecast_category_edges,
        dim,
    ):
        self.observations = observations
        self.forecasts = forecasts
        self.observation_category_edges = observation_category_edges
        self.forecast_category_edges = forecast_category_edges
        self.dichotomous = (
            True
            if (len(observation_category_edges) - 1 == 2)
            & (len(forecast_category_edges) - 1 == 2)
            else False
        )
        self.table = self._get_contingency_table(dim)

    def _get_contingency_table(self, dim):
        table = histogram(
            self.observations,
            self.forecasts,
            bins=[self.observation_category_edges, self.forecast_category_edges],
            bin_names=[OBSERVATIONS_NAME, FORECASTS_NAME],
            dim=dim,
            bin_dim_suffix='_bin',
        )

        # Add some coordinates to simplify interpretation/post-processing
        table = table.assign_coords(
            {
                OBSERVATIONS_NAME
                + '_bin': get_category_bounds(self.observation_category_edges)
            }
        ).rename({OBSERVATIONS_NAME + '_bin': OBSERVATIONS_NAME + '_category_bounds'})
        table = table.assign_coords(
            {FORECASTS_NAME + '_bin': get_category_bounds(self.forecast_category_edges)}
        ).rename({FORECASTS_NAME + '_bin': FORECASTS_NAME + '_category_bounds'})
        table = table.assign_coords(
            {
                OBSERVATIONS_NAME
                + '_category': (
                    OBSERVATIONS_NAME + '_category_bounds',
                    range(1, len(self.observation_category_edges)),
                ),
                FORECASTS_NAME
                + '_category': (
                    FORECASTS_NAME + '_category_bounds',
                    range(1, len(self.forecast_category_edges)),
                ),
            }
        )
        table = table.swap_dims(
            {
                OBSERVATIONS_NAME + '_category_bounds': OBSERVATIONS_NAME + '_category',
                FORECASTS_NAME + '_category_bounds': FORECASTS_NAME + '_category',
            }
        )

        return table

    def _sum_categories(self, categories):
        """
            Returns sums of specified categories in contingency table

            Parameters
            ----------
            category : str, optional
                Contingency table categories to sum. Options are 'total', 'observations' and 'forecasts'

            Returns
            -------
            Sum of all counts in specified categories

        """

        if categories == 'total':
            N = self.table.sum(
                dim=(OBSERVATIONS_NAME + '_category', FORECASTS_NAME + '_category'),
                skipna=True,
            )
        elif categories == 'observations':
            N = self.table.sum(dim=FORECASTS_NAME + '_category', skipna=True).rename(
                {OBSERVATIONS_NAME + '_category': 'category'}
            )
        elif categories == 'forecasts':
            N = self.table.sum(dim=OBSERVATIONS_NAME + '_category', skipna=True).rename(
                {FORECASTS_NAME + '_category': 'category'}
            )
        else:
            raise ValueError(f'"{categories}" is not a recognised category')

        return N

    def __repr__(self):
        return _display_metadata(self)

    @dichotomous_only
    def bias_score(self, yes_category=2):
        """
            Returns the bias score(s) for dichotomous contingency data

            Parameters
            ----------
            yes_category : value, optional
                The category coordinate value of the category corresponding to 'yes'

            Returns
            -------
            xarray.Dataset or xarray.DataArray
                An array containing the bias score(s)

            Notes
            -----
            See http://www.cawcr.gov.au/projects/verification/
        """

        no_category = abs(yes_category - 2) + 1

        hits = self.table.sel(
            {
                OBSERVATIONS_NAME + '_category': yes_category,
                FORECASTS_NAME + '_category': yes_category,
            },
            drop=True,
        )
        false_alarms = self.table.sel(
            {
                OBSERVATIONS_NAME + '_category': no_category,
                FORECASTS_NAME + '_category': yes_category,
            },
            drop=True,
        )
        misses = self.table.sel(
            {
                OBSERVATIONS_NAME + '_category': yes_category,
                FORECASTS_NAME + '_category': no_category,
            },
            drop=True,
        )

        return (hits + false_alarms) / (hits + misses)

    @dichotomous_only
    def hit_rate(self, yes_category=2):
        """
            Returns the hit rate(s) (probability of detection) for dichotomous contingency data

            Parameters
            ----------
            yes_category : value, optional
                The category coordinate value of the category corresponding to 'yes' (1 or 2)

            Returns
            -------
            xarray.Dataset or xarray.DataArray
                An array containing the hit rate(s)

            Notes
            -----
            See http://www.cawcr.gov.au/projects/verification/
        """

        no_category = abs(yes_category - 2) + 1

        hits = self.table.sel(
            {
                OBSERVATIONS_NAME + '_category': yes_category,
                FORECASTS_NAME + '_category': yes_category,
            },
            drop=True,
        )
        misses = self.table.sel(
            {
                OBSERVATIONS_NAME + '_category': yes_category,
                FORECASTS_NAME + '_category': no_category,
            },
            drop=True,
        )

        return hits / (hits + misses)

    @dichotomous_only
    def false_alarm_ratio(self, yes_category=2):
        """
            Returns the false alarm ratio(s) for dichotomous contingency data

            Parameters
            ----------
            yes_category : value, optional
                The category coordinate value of the category corresponding to 'yes'

            Returns
            -------
            xarray.Dataset or xarray.DataArray
                An array containing the false alarm ratio(s)

            Notes
            -----
            See http://www.cawcr.gov.au/projects/verification/
        """

        no_category = abs(yes_category - 2) + 1

        hits = self.table.sel(
            {
                OBSERVATIONS_NAME + '_category': yes_category,
                FORECASTS_NAME + '_category': yes_category,
            },
            drop=True,
        )
        false_alarms = self.table.sel(
            {
                OBSERVATIONS_NAME + '_category': no_category,
                FORECASTS_NAME + '_category': yes_category,
            },
            drop=True,
        )

        return false_alarms / (hits + false_alarms)

    @dichotomous_only
    def false_alarm_rate(self, yes_category=2):
        """
            Returns the false alarm rate(s) for dichotomous contingency data

            Parameters
            ----------
            yes_category : value, optional
                The category coordinate value of the category corresponding to 'yes'

            Returns
            -------
            xarray.Dataset or xarray.DataArray
                An array containing the false alarm rate(s)

            Notes
            -----
            See http://www.cawcr.gov.au/projects/verification/
        """

        no_category = abs(yes_category - 2) + 1

        false_alarms = self.table.sel(
            {
                OBSERVATIONS_NAME + '_category': no_category,
                FORECASTS_NAME + '_category': yes_category,
            },
            drop=True,
        )
        correct_negs = self.table.sel(
            {
                OBSERVATIONS_NAME + '_category': no_category,
                FORECASTS_NAME + '_category': no_category,
            },
            drop=True,
        )

        return false_alarms / (correct_negs + false_alarms)

    @dichotomous_only
    def success_ratio(self, yes_category=2):
        """
            Returns the success ratio(s) for dichotomous contingency data

            Parameters
            ----------
            yes_category : value, optional
                The category coordinate value of the category corresponding to 'yes'

            Returns
            -------
            xarray.Dataset or xarray.DataArray
                An array containing the success ratio(s)

            Notes
            -----
            See http://www.cawcr.gov.au/projects/verification/
        """

        no_category = abs(yes_category - 2) + 1

        hits = self.table.sel(
            {
                OBSERVATIONS_NAME + '_category': yes_category,
                FORECASTS_NAME + '_category': yes_category,
            },
            drop=True,
        )
        false_alarms = self.table.sel(
            {
                OBSERVATIONS_NAME + '_category': no_category,
                FORECASTS_NAME + '_category': yes_category,
            },
            drop=True,
        )

        return hits / (hits + false_alarms)

    @dichotomous_only
    def threat_score(self, yes_category=2):
        """
            Returns the threat score(s) for dichotomous contingency data

            Parameters
            ----------
            yes_category : value, optional
                The category coordinate value of the category corresponding to 'yes'

            Returns
            -------
            xarray.Dataset or xarray.DataArray
                An array containing the threat score(s)

            Notes
            -----
            See http://www.cawcr.gov.au/projects/verification/
        """

        no_category = abs(yes_category - 2) + 1

        hits = self.table.sel(
            {
                OBSERVATIONS_NAME + '_category': yes_category,
                FORECASTS_NAME + '_category': yes_category,
            },
            drop=True,
        )
        misses = self.table.sel(
            {
                OBSERVATIONS_NAME + '_category': yes_category,
                FORECASTS_NAME + '_category': no_category,
            },
            drop=True,
        )
        false_alarms = self.table.sel(
            {
                OBSERVATIONS_NAME + '_category': no_category,
                FORECASTS_NAME + '_category': yes_category,
            },
            drop=True,
        )

        return hits / (hits + misses + false_alarms)

    @dichotomous_only
    def equit_threat_score(self, yes_category=2):
        """
            Returns the equitable threat score(s) for dichotomous contingency data

            Parameters
            ----------
            yes_category : value, optional
                The category coordinate value of the category corresponding to 'yes'

            Returns
            -------
            xarray.Dataset or xarray.DataArray
                An array containing the equitable threat score(s)

            Notes
            -----
            See http://www.cawcr.gov.au/projects/verification/
        """

        no_category = abs(yes_category - 2) + 1

        hits = self.table.sel(
            {
                OBSERVATIONS_NAME + '_category': yes_category,
                FORECASTS_NAME + '_category': yes_category,
            },
            drop=True,
        )
        misses = self.table.sel(
            {
                OBSERVATIONS_NAME + '_category': yes_category,
                FORECASTS_NAME + '_category': no_category,
            },
            drop=True,
        )
        false_alarms = self.table.sel(
            {
                OBSERVATIONS_NAME + '_category': no_category,
                FORECASTS_NAME + '_category': yes_category,
            },
            drop=True,
        )
        hits_random = ((hits + misses) * (hits + false_alarms)) / self._sum_categories(
            'total'
        )

        return (hits - hits_random) / (hits + misses + false_alarms - hits_random)

    @dichotomous_only
    def odds_ratio(self, yes_category=2):
        """
            Returns the odds ratio(s) for dichotomous contingency data

            Parameters
            ----------
            yes_category : value, optional
                The category coordinate value of the category corresponding to 'yes'

            Returns
            -------
            xarray.Dataset or xarray.DataArray
                An array containing the equitable odds ratio(s)

            Notes
            -----
            See http://www.cawcr.gov.au/projects/verification/
        """

        no_category = abs(yes_category - 2) + 1

        hits = self.table.sel(
            {
                OBSERVATIONS_NAME + '_category': yes_category,
                FORECASTS_NAME + '_category': yes_category,
            },
            drop=True,
        )
        correct_negs = self.table.sel(
            {
                OBSERVATIONS_NAME + '_category': no_category,
                FORECASTS_NAME + '_category': no_category,
            },
            drop=True,
        )
        misses = self.table.sel(
            {
                OBSERVATIONS_NAME + '_category': yes_category,
                FORECASTS_NAME + '_category': no_category,
            },
            drop=True,
        )
        false_alarms = self.table.sel(
            {
                OBSERVATIONS_NAME + '_category': no_category,
                FORECASTS_NAME + '_category': yes_category,
            },
            drop=True,
        )

        return (hits * correct_negs) / (misses * false_alarms)

    @dichotomous_only
    def odds_ratio_skill_score(self, yes_category=2):
        """
            Returns the odds ratio skill score(s) for dichotomous contingency data

            Parameters
            ----------
            yes_category : value, optional
                The category coordinate value of the category corresponding to 'yes'

            Returns
            -------
            xarray.Dataset or xarray.DataArray
                An array containing the equitable odds ratio skill score(s)

            Notes
            -----
            See http://www.cawcr.gov.au/projects/verification/
        """

        no_category = abs(yes_category - 2) + 1

        hits = self.table.sel(
            {
                OBSERVATIONS_NAME + '_category': yes_category,
                FORECASTS_NAME + '_category': yes_category,
            },
            drop=True,
        )
        correct_negs = self.table.sel(
            {
                OBSERVATIONS_NAME + '_category': no_category,
                FORECASTS_NAME + '_category': no_category,
            },
            drop=True,
        )
        misses = self.table.sel(
            {
                OBSERVATIONS_NAME + '_category': yes_category,
                FORECASTS_NAME + '_category': no_category,
            },
            drop=True,
        )
        false_alarms = self.table.sel(
            {
                OBSERVATIONS_NAME + '_category': no_category,
                FORECASTS_NAME + '_category': yes_category,
            },
            drop=True,
        )

        return (hits * correct_negs - misses * false_alarms) / (
            hits * correct_negs + misses * false_alarms
        )

    def accuracy(self):
        """
            Returns the accuracy score(s) for a contingency table

            Returns
            -------
            xarray.Dataset or xarray.DataArray
                An array containing the accuracy score(s)

            Notes
            -----
            See http://www.cawcr.gov.au/projects/verification/
        """

        hits = self.table.where(
            self.table[OBSERVATIONS_NAME + '_category']
            == self.table[FORECASTS_NAME + '_category']
        ).sum(
            dim=(OBSERVATIONS_NAME + '_category', FORECASTS_NAME + '_category'),
            skipna=True,
        )
        N = self._sum_categories('total')

        return hits / N

    def Heidke_score(self):
        """
            Returns the Heidke skill score(s) for a contingency table

            Returns
            -------
            xarray.Dataset or xarray.DataArray
                An array containing the Heidke score(s)

            Notes
            -----
            See http://www.cawcr.gov.au/projects/verification/
        """

        numer_1 = self.table.where(
            self.table[OBSERVATIONS_NAME + '_category']
            == self.table[FORECASTS_NAME + '_category']
        ).sum(
            dim=(OBSERVATIONS_NAME + '_category', FORECASTS_NAME + '_category'),
            skipna=True,
        ) / self._sum_categories(
            'total'
        )
        numer_2 = (
            self._sum_categories('observations') * self._sum_categories('forecasts')
        ).sum(dim='category', skipna=True) / self._sum_categories('total') ** 2
        denom = 1 - numer_2

        return (numer_1 - numer_2) / denom

    def Peirce_score(self):
        """
            Returns the Peirce skill score(s) (Hanssen and Kuipers discriminant) for a \
                contingency table

            Returns
            -------
            xarray.Dataset or xarray.DataArray
                An array containing the Peirce score(s)

            Notes
            -----
            See http://www.cawcr.gov.au/projects/verification/
        """

        numer_1 = self.table.where(
            self.table[OBSERVATIONS_NAME + '_category']
            == self.table[FORECASTS_NAME + '_category']
        ).sum(
            dim=(OBSERVATIONS_NAME + '_category', FORECASTS_NAME + '_category'),
            skipna=True,
        ) / self._sum_categories(
            'total'
        )
        numer_2 = (
            self._sum_categories('observations') * self._sum_categories('forecasts')
        ).sum(dim='category', skipna=True) / self._sum_categories('total') ** 2
        denom = 1 - (self._sum_categories('observations') ** 2).sum(
            dim='category', skipna=True
        ) / (self._sum_categories('total') ** 2)

        return (numer_1 - numer_2) / denom

    def Gerrity_score(self):
        """
            Returns Gerrity equitable score for a contingency table

            Returns
            -------
            xarray.Dataset or xarray.DataArray
                An array containing the Gerrity scores

            Notes
            -----
            See http://www.cawcr.gov.au/projects/verification/

            To do

            - Currently computes the Gerrity scoring matrix using nested for-loops. Is it possible \
                    to remove these?
        """

        def _Gerrity_s(table):
            """
                Returns Gerrity scoring matrix, s
            """
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
            _Gerrity_s,
            self.table,
            input_core_dims=[
                [OBSERVATIONS_NAME + '_category', FORECASTS_NAME + '_category']
            ],
            output_core_dims=[
                [OBSERVATIONS_NAME + '_category', FORECASTS_NAME + '_category']
            ],
            dask='allowed',
        )

        return (self.table * s).sum(
            dim=(OBSERVATIONS_NAME + '_category', FORECASTS_NAME + '_category'),
            skipna=True,
        ) / self._sum_categories('total')
