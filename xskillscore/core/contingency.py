import xarray as xr
from xhistogram.xarray import histogram as xhist

__all__ = [
    'histogram'
    'contingency_table',
    'hit_rate'
]

def histogram(*args, bins=None, bin_names=None, **kwargs):
    """
        Wrapper on xhistogram to deal with DataSets
    """
    if isinstance(args[0], xr.core.dataset.Dataset):
        # Get list of variables that are shared across all Datasets
        overlapping_vars = set.intersection(*map(set,[arg.data_vars for arg in args]))
        if overlapping_vars:
            # If bin_names not provided, use default ----
            if bin_names is None:
                bin_names = ['ds_'+str(i+1) for i in range(len(args))]
            return xr.merge([xhist(*(arg[var].rename(bin_names[i]) for i, arg in enumerate(args)), 
                                   bins=bins, **kwargs).rename(var)
                             for var in overlapping_vars])
        else:
            raise ValueError('No common variables exist between DataSets a and b')
    else:
        return xhist(*args, bins=bins, **kwargs)

    
def contingency_table(a, b, a_category_edges, b_category_edges, dim, 
                      a_category_name='a', b_category_name='b'):
    """ 
        Return the contingency table between a and b for given categories
        
        Parameters
        ----------
        a : xarray.Dataset or xarray.DataArray
            Labeled array(s) over which to apply the function.
        b : xarray.Dataset or xarray.DataArray
            Labeled array(s) over which to apply the function.
        category_edges_a : array_like
            Bin edges for categorising a
        category_edges_b : array_like
            Bin edges for categorising b
        dim : str, list
            The dimension(s) over which to compute the contingency table
            
        Returns
        -------
        xarray.Dataset or xarray.DataArray
            Contingency table of input data
            
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
        >>> contingency_table(a, b, category_edges_a, category_edges_b, dim=['x','y'])
        <xarray.Dataset>
        Dimensions:  (a_bin: 4, b_bin: 4)
        Coordinates:
          * a_bin    (a_bin) float64 -1.5 -0.5 0.5 1.5
          * b_bin    (b_bin) float64 -2.25 -0.75 0.75 2.25
        Data variables:
            test2    (a_bin, b_bin) int64 0 1 1 0 1 0 1 0 0 1 1 1 0 1 0 0
            test1    (a_bin, b_bin) int64 0 1 1 0 0 1 1 1 0 0 1 0 0 1 1 0
        
        Notes
        -----
        See http://www.cawcr.gov.au/projects/verification/
    """
    
    contingency = histogram(a, b, 
                            bins=[a_category_edges, b_category_edges], 
                            bin_names=[a_category_name, b_category_name],
                            dim=dim, bin_dim_suffix='_bin')
    
    # Add some coordinates to simplify interpretation/post-processing
    contingency = contingency.assign_coords({a_category_name+'_category': (a_category_name+'_bin', 
                                                                           range(1,len(a_category_edges))),
                                             b_category_name+'_category': (b_category_name+'_bin', 
                                                                           range(1,len(b_category_edges)))})
    a_category_bounds = ['('+str(a_category_edges[i])+', '+str(a_category_edges[i+1])+']' 
                         for i in range(len(a_category_edges)-1)]
    b_category_bounds = ['('+str(b_category_edges[i])+', '+str(b_category_edges[i+1])+']' 
                         for i in range(len(b_category_edges)-1)]
    contingency = contingency.assign_coords({a_category_name+'_bin_edges': (a_category_name+'_bin', 
                                                                            a_category_bounds),
                                             b_category_name+'_bin_edges': (b_category_name+'_bin', 
                                                                            b_category_bounds)})
    contingency = contingency.swap_dims({a_category_name+'_bin': a_category_name+'_category',
                                         b_category_name+'_bin': b_category_name+'_category'})
    
    return contingency


def hit_rate(contingency, forecast_category_name, reference_category_name, yes_category=2):
    """ 
        Returns the hit rate (probability of detection) given dichotomous contingency data 
        
        Parameters
        ----------
        contingency : xarray.Dataset or xarray.DataArray
            A 2 category contingency table of the form output from contingency.contingency_table
        yes_category : value, optional
            The coordinate value of the category corresponding to 'yes' (1 or 2)
            
        Returns
        -------
        xarray.Dataset or xarray.DataArray
            An array containing the hit rates
            
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
        >>> category_edges_a = np.linspace(-2,2,3)
        >>> category_edges_b = np.linspace(-2,2,3)
        >>> contingency = contingency_table(a, b, category_edges_a, category_edges_b, dim=['x','y'])
        >>> hit_rate(contingency, forecast_category_name='a_category', reference_category_name='b_category')
        <xarray.Dataset>
        Dimensions:  ()
        Data variables:
            test2    float64 0.5
            test1    float64 0.0
          
        Notes
        -----
        See http://www.cawcr.gov.au/projects/verification/
    """
    
    no_category = abs(yes_category - 2) + 1
    
    if len(contingency[forecast_category_name]) > 2:
        raise ValueError('Hit rate is defined for dichotomous contingency data only')
    
    hits = contingency.sel({forecast_category_name: yes_category, 
                            reference_category_name: yes_category}, drop=True)
    misses = contingency.sel({forecast_category_name: no_category, 
                              reference_category_name: yes_category}, drop=True)
    
    return hits / (hits + misses)