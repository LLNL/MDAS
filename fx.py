#!/bin/env python
# -*- coding: utf-8 -*-

"""
Stephen Po-Chedley 23 August 2022

Function library used to disentangle the forced and unforced
components of tropospheric warming.

@author: pochedls
"""

# imports
import numpy as np
import re
from sklearn.neural_network import MLPRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from fx_odr import orthoregress
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def natural_sort(l):
    """ l_sorted = natural_sort(l)

    Sort list using intuitive alpha-numeric ordering.

    """
    # better sorting of text lists
    # https://stackoverflow.com/questions/4836710/is-there-a-built-in-function-for-string-natural-sort
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def get_lat_lon_weight():
    """
    lat, lon, weights = get_lat_lon_weight()

    Function returns the latitude, longitude, and spatial weights.

    Returns
    -------
    lat : np.array
        Array of latitude values

    lon : np.array
        Array of longitude values

    weights : np.array
        Array of cosine(lat) weights [1, nlat*nlon]

    Notes
    ---------
    The returned values correspond to a 2.5 x 2.5 degree grid.

    """
    # specify lat/lon
    lat = np.arange(-88.75, 90, 2.5)
    lon = np.arange(1.25, 360, 2.5)
    # %% Create weight vector
    weight = np.cos(np.radians(lat))
    weight = np.tile(np.expand_dims(weight, axis=1), (1, len(lon)))
    weight = np.reshape(weight, (1, len(lat)*len(lon)))

    return lat, lon, weight


def get_training_matrix(trends, trendMaps, leave_out_models=[], leave_out_periods=[], record_len=36, n_members=10, ptype=['total', 'forced', 'unforced']):
    """
    X, Y = get_training_matrix(trends, trendMaps)

    Function takes a dictionary of trend values and trend maps
    and returns predictor and predictand matrices. The predictor
    matrix is composed of trend maps and the predictand matrix is
    composed of trend values.


    Parameters
    ----------
    trends : Dict
        A nested dictionary composed of trend values. The dictionary
        has the form trends[model][member][period]. The climate model and 
        ensemble member are specified with a string and the period is a tuple of
        int years (start, end). The end year is inclusive (i.e., goes
        through December 31). Each entry is a list of trends in the following
        order: total, forced, unforced.

    trendMaps : Dict
        A nested dictionary composted of trend maps. The dictionary
        has the form trendMaps[model][member][period] (as with trends).

    leave_out_models : list, optional
        A list of climate models to leave out of the predictor and predictand
        matrices. The default is an empty list.

    leave_out_periods : list, optional
        A list of periods to leave out of the predictor and predictand
        matrices. The default is an empty list.

    record_len : int, optional
        Length of record to be analyzed (in years); other periods are
        ignored. The default is 36.

    n_members : int, optional
        Number of members to be considered from each climate model for training.
        The default is 10.

    ptype : list, optional
        List of trend types to be considered (a combination of 'forced',
        'unforced', and 'total').

    Returns
    -------
    X : np.array
        Array of predictors (nsamples, nlat*nlon)

    Y : np.array
        Array of predictands (nsamples, n_ptypes)

    """
    # hardcode trend dictionary matrix
    trendDictMapping = {'total': 0, 'forced': 1, 'unforced': 2}
    # get climate models
    models = [model for model in trends.keys() if model not in leave_out_models]
    # initialize training matrices
    X = []
    Y = []
    # loop over climate models to get training matrix
    for model in models:
        # order members
        members = natural_sort(trends[model].keys())
        # choose periods of correct length
        # and leave out specified periods
        all_periods = list(trends[model][members[0]].keys())
        periods = []
        for period in all_periods:
            if period in leave_out_periods:
                continue
            if period[1] - period[0] + 1 != record_len:
                continue
            periods.append(period)
        # loop over members and get data (up to n_members)
        model_count = 0
        XM = []
        YM = []
        for member in members:
            if model_count >= n_members:
                continue
            # check if data is missing from member
            if len(periods) == 0:
                continue
            if periods[0] not in trendMaps[model][member].keys():
                continue
            # increment model_count
            model_count += 1
            # append data to training matrix
            for period in periods:
                yrow = trends[model][member][period]
                xrow = np.reshape(trendMaps[model][member][period], -1)
                XM.append(xrow)
                YM.append(yrow)
        # if model had sufficient data, add it to array
        if model_count == n_members:
            X = X + XM
            Y = Y + YM
    # cast to numpy array
    X = np.array(X)
    Y = np.array(Y)
    # choose specified predictands
    inds = [trendDictMapping[p] for p in ptype]
    Y = Y[:, inds]

    return X, Y


def standardize(X, Y, weight=None):
    """
    X, Y, XM, XS, YM, YS = standardize(X, Y, weight=weights)

    Function standardizes each data point in the X and Y matrix such that
    the mean is zero and the standard deviation is one.

    Parameters
    ----------
    X : np.array
        Array of predictors (nsamples, nlat*nlon)

    Y : np.array
        Array of predictands (nsamples, n_ptypes)

    weight : np.array, optional
        Matrix of weights (e.g., spatial) that can be multiplied with the
        predictor matrix (if included, the default is None).

    Returns
    -------
    X : np.array
        Array of predictors after standardization (nsamples, nlat*nlon)

    Y : np.array
        Array of predictands after standardization (nsamples, n_ptypes)

    XM : np.array
        Mean of original predictors at each grid cell (nlat*nlon)

    XS : np.array
        Standard deviation of original predictors at each grid cell
        (nlat*nlon)

    YM : np.array
        Mean of original predictands over all samples (n_ptypes)

    YS : np.array
        Standard deviation of original predictands over all samples (n_ptypes)

    """
    # get coefficients
    XM = np.mean(X, axis=0)
    XS = np.std(X, axis=0, ddof=1)
    YM = np.mean(Y, axis=0)
    YS = np.std(Y, axis=0, ddof=1)
    # scale
    X = (X - XM) / XS
    Y = (Y - YM) / YS
    # apply weights if specified
    if weight is not None:
        w = np.tile(weight, (X.shape[0], 1))
        X = X*w
    return X, Y, XM, XS, YM, YS


def get_model_data(trends, trendMaps, model, ptype=['unforced', 'forced']):
    """
    XP, YP, fit_labels = get_model_data(trends, trendMaps, model)

    Function retrieves the predictor maps and predictand trend values for
    a specified climate model.

    Parameters
    ----------
    trends : Dict
        A nested dictionary composted of trend values. The dictionary
        has the form trends[model][member][period]. The climate model and 
        ensemble member are specified with a string and the period is a tuple of
        int years (start, end). The end year is inclusive (i.e., goes
        through December 31). Each entry is a list of trends in the following
        order: total, forced, unforced.

    trendMaps : Dict
        A nested dictionary composted of trend maps. The dictionary
        has the form trendMaps[model][member][period] (as with trends).

    ptype : list, optional
        List of trend types to be retrieved (a combination of 'forced',
        'unforced', and 'total'). Default is ['unforced', 'forced'].

    Returns
    -------
    X : np.array
        Array of predictors (nsamples, nlat*nlon)

    Y : np.array
        Array of predictands (nsamples, n_ptypes)

    fit_labels : list
        List labeling each sample. Each entry includes:
        [model, member, period]

    """
    # hardcode trend dictionary matrix
    trendDictMapping = {'total': 0, 'forced': 1, 'unforced': 2}
    # loop over and get members
    members = list(trends[model].keys())
    periods = list(trends[model][members[0]].keys())
    XP = []
    YP = []
    fit_labels = []
    for member in members:
        for period in periods:
            if period not in trendMaps[model][member].keys():
                continue
            YP.append(trends[model][member][period])
            XP.append(np.reshape(trendMaps[model][member][period], -1))
            fit_labels.append([model, member, period])
    XP = np.array(XP)
    YP = np.array(YP)
    # choose specified predictands
    inds = [trendDictMapping[p] for p in ptype]
    YP = YP[:, inds]
    return XP, YP, fit_labels


def apply_predictor_scaling(X, XM, XS, weight=None):
    """
    X = apply_predictor_scaling(X, XM, XS)

    Function standardizes each data point in the predictand (X) matrix using
    specified scaling coefficients such that:
        X = (X - XM) / XS

    Parameters
    ----------
    X : np.array
        Array of predictors (nsamples, nlat*nlon)

    XM : np.array
        Array to subtract from input matrix (nlat*nlon)

    XS : np.array
        Array to divide from input matrix (nlat*nlon)

    weight : np.array, optional
        Matrix of weights (e.g., spatial) that can be multiplied with the
        predictor matrix (if included, the default is None).

    Returns
    -------
    X : np.array
        Array of predictors after standardization (nsamples, nlat*nlon)
    """
    # scale
    X = (X - XM) / XS
    # apply weights if specified
    if weight is not None:
        w = np.tile(weight, (X.shape[0], 1))
        X = X*w

    return X


def unscale_predictands(yp, YM, YS):
    """
    YU = unscale_predictands(Y, YM, YS)

    Function removes standardization for each data point in the predictand (y) matrix
    using specified scaling coefficients such that:
        YU = Y*YS + YM

    Parameters
    ----------
    Y : np.array
        Array of standardized predictands (nsamples, n_ptypes)

    YM : np.array
        Array of values to be added to input predictands (n_ptypes)

    YS : np.array
        Array of values that are multiplied with the input predictands (n_ptypes)

    Returns
    -------
    YU : np.array
        Array of predictors after standardization is removed (nsamples, nlat*nlon)
    """
    return yp*YS+YM


def initialize_ml_model(model_type, **kwargs):
    """
    mlmodel, ml_args = initialize_ml_model(model_type, **kwargs)

    Function initializes a scikit-learn machine learning model.

    Parameters
    ----------
    model_type : str
        Model type to initialize. Can specify:
            - 'nn': MLPRegressor neural network
            - 'ridge': Ridge regression
            - 'pls': PLS regression

    kwargs : dict, optional
        Dictionary of model specifications. These can be in addition to
        the defaults (see Notes) or can be specified to overwrite the
        default values.

    Returns
    -------
    mlmodel : sklearn.MODEL
        Initialized scikit-learn machine learning model.

    ml_args : dict
        Parameters specified in producing mlmodel.

    Notes
    -----
    A number of default arguments are supplied for each model. They are:

    Ridge:
        - alpha: 1
        - fit_intercept: False
    PLS:
        - n_components: 3
        - fit_intercept: False
    NN:
        - hidden_layer_sizes: (10, 10)
        - activation: relu
        - solver: sgd
        - alpha: 1
        - max_iter: 2000
        - shuffle: False
        - verbose: True
        - warm_start: False
        - random_state: 0
        - validation_fraction: 0.2
        - early_stopping: True

    scikit-learn model documentation:
        - NN: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
        - Ridge: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
        - PLS: https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html

    """
    # define default model parameters
    default_ridge = {'alpha': 1, 'fit_intercept': False}
    default_pls = {'n_components': 3, 'scale': False}
    default_nn = {'hidden_layer_sizes': (10, 10),
                  'activation': 'relu',
                  'solver': 'sgd',
                  'alpha': 1,
                  'max_iter': 2000,
                  'shuffle': False,
                  'verbose': True,
                  'warm_start': False,
                  'random_state': 0,
                  'validation_fraction': 0.2,
                  'early_stopping': True}
    ml_args = {'ridge': default_ridge,
               'pls': default_pls,
               'nn': default_nn}
    # choose model default parameters based on specified model type
    ml_args = ml_args[model_type]
    # update model parameters based on user optional arguments
    for key in kwargs.keys():
        ml_args[key] = kwargs[key]
    # create ML model instance
    if model_type == 'pls':
        mlmodel = PLSRegression(**ml_args)
    elif model_type == 'ridge':
        mlmodel = Ridge(**ml_args)
    elif model_type == 'nn':
        mlmodel = MLPRegressor(**ml_args)
    # return model and model arguments
    return mlmodel, ml_args


def predictions_to_dict(trendPredict, YP, fit_labels, ptype=['unforced', 'forced']):
    """
    trendPredict = predictions_to_dict(trendPredict, YP, fit_labels, ptype=['unforced', 'forced'])

    Function updates a dictionary with trend predictions.

    Parameters
    ----------
    trendPredict : dict
        A nested dictionary composed of trend values. The dictionary
        has the form trends[model][member][period]. The climate model and 
        ensemble member are specified with a string and the period is a tuple of
        int years (start, end). The end year is inclusive (i.e., goes
        through December 31). Each entry is a list of trends stored in the 
        following order: total, forced, unforced.

    YP : np.array
        Array of trend predictions (nsamples, n_ptype)

    fit_labels : list
        List labeling each sample. Each entry includes:
        [model, member, period]

    ptype : list, optional
        List of trend types used during model fitting (a combination
        of 'forced', 'unforced', and 'total'). Default assumes
        ['unforced', 'forced'].

    Returns
    -------
    trendPredict : dict
        Updated dictionary.

    """
    # hardcode trend dictionary matrix
    trendDictMapping = {'total': 0, 'forced': 1, 'unforced': 2}
    # include total in output as summed unforced + forced trend
    # if not specified as a predictand
    if (('total' not in ptype) & ('unforced' in ptype) & ('forced' in ptype)):
        YP = np.concatenate((YP, np.expand_dims(YP[:, 0] + YP[:, 1], axis=1)), axis=1)
        ptype.append('total')
    # loop over each fit label in list
    for i, (model, member, period) in enumerate(fit_labels):
        # initialize a values array
        values = np.array([np.nan]*3)
        # initialize climate model nested dict if need be
        if model not in trendPredict.keys():
            trendPredict[model] = {}
        # initialize ensemble member nested dict if need be
        if member not in trendPredict[model].keys():
            trendPredict[model][member] = {}
        # get indices for mapping
        inds = [trendDictMapping[p] for p in ptype]
        # place values into array
        values[inds] = YP[i]
        # store array in dictionary
        trendPredict[model][member][period] = values

    return trendPredict


def observed_predictions_to_dict(trendPredict, YP, model, fit_labels, ptype=['unforced', 'forced']):
    """
    trendPredict = observed_predictions_to_dict(trendPredict, YP, model, fit_labels, ptype=['unforced', 'forced'])

    Function updates a dictionary with trend predictions for observations.

    Parameters
    ----------
    trendPredict : dict
        A nested dictionary composed of trend values. The dictionary
        has the form trends[model][ds][member][period]. The climate model, dataset
        (ds), and ensemble member are specified with a string and the period is a
        tuple of int years (start, end). The end year is inclusive (i.e., goes
        through December 31). Each entry is a list of trends stored in the 
        following order: total, forced, unforced.

    YP : np.array
        Array of trend predictions (nsamples, n_ptype).

    model : str
        Climate model name for labelling results in dictionary. Note that in this
        implementation this refers to the model left out of the fitting process.

    fit_labels : list
        List labeling each sample. Each entry includes:
        [dataset, member, period]

    ptype : list, optional
        List of trend types used during model fitting (a combination
        of 'forced', 'unforced', and 'total'). Default assumes
        ['unforced', 'forced'].

    Returns
    -------
    trendPredict : dict
        Updated dictionary.

    """
    # hardcode trend dictionary matrix
    trendDictMapping = {'total': 0, 'forced': 1, 'unforced': 2}
    # initialize nested dictionary for climate model
    trendPredict[model] = {}
    # include total in output as summed unforced + forced trend
    # if not specified as a predictand
    if (('total' not in ptype) & ('unforced' in ptype) & ('forced' in ptype)):
        YP = np.concatenate((YP, np.expand_dims(YP[:, 0] + YP[:, 1], axis=1)), axis=1)
        ptype.append('total')
    # loop over each fit label in list
    for i, (ds, member, period) in enumerate(fit_labels):
        # initialize a values array
        values = np.array([np.nan]*3)
        # initialize dataset nested dict if need be
        if ds not in trendPredict[model].keys():
            trendPredict[model][ds] = {}
        # initialize ensemble member nested dict if need be
        if member not in trendPredict[model][ds].keys():
            trendPredict[model][ds][member] = {}
        # get indices for mapping
        inds = [trendDictMapping[p] for p in ptype]
        # place values into array
        values[inds] = YP[i]
        # store array in dictionary
        trendPredict[model][ds][member][period] = values

    return trendPredict


def store_coefficients(ml_coefs, model, model_type, mlmodel, shp, data_inds=None):
    """
    ml_coefs = store_coefficients(ml_coefs, model, model_type, mlmodel, shp)

    Store scikit-learn fitted model coefficients in a nested dictionary.

    Parameters
    ----------
    ml_coefs : dict
        A nested dictionary composed of the fit coefficients in the form of
        ml_coefs[coefficient][model] where the coefficient and climate model
        are both strings. The coefficient values vary by scikit-learn model
        type. A neural network ('nn') has no coefficients. Ridge and PLS
        regression have the following coefficients:
            - Ridge:    beta
            - PLS:      beta
                        x_loadings_
                        y_loadings_
                        x_scores_
                        y_scores_
                        x_weights_
                        y_weights_

    model : str
        Name of climate model that pertains to scikit-learn model (in our case
        the model left out of training).

    model_type : str
        Model type used. Can specify:
            - 'nn': MLPRegressor neural network
            - 'ridge': Ridge regression
            - 'pls': PLS regression

    mlmodel : sklearn.MODEL
        Fitted scikit-learn machine learning model.

    shp : tuple
        Two-element tuple of the grid shape (nlat, nlon).

    Returns
    -------
    ml_coefs : dict
        Updated dictionary.

    """
    # get model coefficients
    if model_type == 'ridge':
        if ml_coefs == {}:
            ml_coefs['beta'] = {}
        beta = mlmodel.coef_
        if data_inds is not None:
            beta_expand = np.zeros((2, shp[0]*shp[1]))
            beta_expand[:, data_inds] = beta
            beta = beta_expand
        beta = np.reshape(beta, (2, shp[0], shp[1]))
        ml_coefs['beta'][model] = beta
        return ml_coefs
    elif model_type == 'pls':
        ncomps = mlmodel.n_components
        if ml_coefs == {}:
            ml_coefs = {'beta': {}, 'x_loadings_': {}, 'y_loadings_': {},
                        'x_scores_': {}, 'y_scores_': {}, 'x_weights_': {},
                        'y_weights_': {}}
        x_loadings_ = mlmodel.x_loadings_.T
        x_weights_ = mlmodel.x_weights_.T
        beta = mlmodel.coef_
        if data_inds is not None:
            beta_expand = np.zeros((shp[0]*shp[1], 2))*np.nan
            beta_expand[data_inds, :] = beta
            beta = beta_expand
            xle = np.zeros((ncomps, shp[0]*shp[1]))*np.nan
            xwe = np.zeros((ncomps, shp[0]*shp[1]))*np.nan
            xle[:, data_inds] = x_loadings_
            xwe[:, data_inds] = x_weights_
            x_weights_ = xwe
            x_loadings_ = xle
        beta = np.reshape(np.transpose(beta, (1, 0)), (2, shp[0], shp[1]))
        x_loadings_ = np.reshape(x_loadings_, (ncomps, shp[0], shp[1]))
        y_loadings_ = mlmodel.y_loadings_.T
        x_scores_ = mlmodel.x_scores_.T
        y_scores_ = mlmodel.y_scores_.T
        x_weights_ = np.reshape(x_weights_, (ncomps, shp[0], shp[1]))
        y_weights_ = mlmodel.y_weights_.T
        # store model coefficents
        ml_coefs['beta'][model] = beta
        ml_coefs['x_loadings_'][model] = x_loadings_
        ml_coefs['y_loadings_'][model] = y_loadings_
        ml_coefs['x_scores_'][model] = x_scores_
        ml_coefs['y_scores_'][model] = y_scores_
        ml_coefs['x_weights_'][model] = x_weights_
        ml_coefs['y_weights_'][model] = y_weights_

        return ml_coefs
    else:

        return ml_coefs


def get_predicted_and_actual_trends(trends, trendPredict, model, period=None, leave_out_periods=[], ptype=['total', 'forced', 'unforced']):
    """
    X, Y = get_predicted_and_actual_trends(trends, trendPredict, model, period=None, leave_out_periods=[], ptype=['total', 'forced', 'unforced'])

    Get predicted trends (X) and actual climate model trends (Y) for a given model.

    Parameters
    ----------
    trends : Dict
        A nested dictionary composed of trend values. The dictionary
        has the form trends[model][member][period]. The climate model and 
        ensemble member are specified with a string and the period is a tuple of
        int years (start, end). The end year is inclusive (i.e., goes
        through December 31). Each entry is a list of trends in the following
        order: total, forced, unforced.

    trendPredict : dict
        A nested dictionary composed of trend values. The dictionary
        has the form trends[model][member][period]. The climate model and 
        ensemble member are specified with a string and the period is a tuple of
        int years (start, end). The end year is inclusive (i.e., goes
        through December 31). Each entry is a list of trends stored in the 
        following order: total, forced, unforced.

    model : str
        Name of climate model that pertains to scikit-learn model (in our case
        the model left out of training).

    period : tuple, optional
        Tuple of the data period to retrieve (start_year, end_year). Note that
        the end_year is inclusive (e.g., goes through December of the end_year).

    leave_out_periods : list, optional
        A list of periods to leave out of the predictor and predictand matrices.
        The default is an empty list. Note that if period is specified, this
        argument is ignored.

    ptype : list, optional
        List of trend types to be considered (a combination of 'forced',
        'unforced', and 'total'). Default is ['total', 'forced', 'unforced'].

    Returns
    -------
    X : np.array
        Predicted values (nsamples, n_ptypes)

    Y : np.array
        Actual model calculated values (nsamples, n_ptypes)        

    Notes
    -----
    Function will get data for all periods unless *either* period or leave_out_periods
    is specified. The number of samples is the product of the number of time periods
    and the number of ensemble members.

    """
    # hardcode trend dictionary matrix
    trendDictMapping = {'total': 0, 'forced': 1, 'unforced': 2}
    # choose specified predictands
    inds = [trendDictMapping[p] for p in ptype]
    # get list of all possible members
    members = list(trends[model].keys())
    # if period is specified, place in list (will loop over one list item);
    # if leave_out_periods are specified, get all periods and then drop the
    # periods that should be left out
    if period is not None:
        periods = [period]
    else:
        periods = list(trendPredict[model][members[0]].keys())
        periods = [period for period in periods if period not in leave_out_periods]
    # initialize output lists
    X = []
    Y = []
    # loop over all ensemble members and specified periods and accumulate
    # predicted and actual trends
    for member in members:
        if member not in trendPredict[model].keys():
            continue
        for period in periods:
            if period not in trendPredict[model][member].keys():
                continue
            x = np.array(trendPredict[model][member][period])[inds]
            y = np.array(trends[model][member][period])[inds]
            X.append(x)
            Y.append(y)
    # cast to array
    X = np.array(X)
    Y = np.array(Y)

    return X, Y


def get_obs_predictor_data(obsTrendMaps, period, infill='zonal', datasets=None):
    """
    X, obs_labels = get_obs_predictor_data(obsTrendMaps, period, infill='zonal')

    Function fetches observational predictor maps for a given time period.

    Parameters
    ----------
    obsTrendMaps : Dict
        A nested dictionary composed of trend values. The dictionary
        has the form trends[model][member][period]. The climate model and 
        ensemble member are specified with a string and the period is a tuple of
        int years (start, end). The end year is inclusive (i.e., goes
        through December 31). Each entry is a list of trends in the following
        order: total, forced, unforced.

    period : tuple
        Tuple of the data period to retrieve (start_year, end_year). Note that
        the end_year is inclusive (e.g., goes through December of the end_year).

    infill : str, optional
        Option to infill missing values. Default is 'zonal'. Options include:
            - 'zero': infill missing data with zeros
            - 'zonal': infill missing data with zonal average (of non-missing data)

    datasets: list, optional
        If populated, only retrieve the listed datasets. If None, retrieve all
        datasets. Default None.

    Returns
    -------
    X : np.array
        Array of predictors (nsamples, nlat*nlon)

    obs_labels : list
        List labeling each sample. Each entry includes:
        [dataset, member, period]

    """
    # initialize output lists
    X = []
    obs_labels = []
    if not datasets:
        datasets = list(obsTrendMaps.keys())
    # loop over observational datasets and dataset members
    for ds in datasets:
        for member in obsTrendMaps[ds].keys():
            if period not in obsTrendMaps[ds][member].keys():
                continue
            # get trend
            m = obsTrendMaps[ds][member][period]
            # infill based on zonal/zero option
            if infill == 'zonal':
                # get zonal mean values
                mz = np.nanmean(m, axis=1)
                # broadcast zonal values to full trend map size
                m, mz = np.broadcast_arrays(m, np.expand_dims(mz, axis=1))
                # place zonal mean values at missing grid cells
                m = np.where(np.isnan(m), mz, m)
            elif infill == 'zero':
                # replace missing values with zeros
                m = np.where(np.isnan(m), 0., m)
            # reshape to nlat*nlon length vector
            m = np.reshape(m, -1)
            # append to predictor matrix
            X.append(m)
            # add label to label list
            obs_labels.append([ds, member, period])
    # cast to numpy array
    X = np.array(X)

    return X, obs_labels


def get_observed_predictions_for_dataset(obsPredict, ds, period, models=None):
    """
    yp = get_observed_predictions_for_dataset(obsPredict, ds, period)

    Function fetches observational predictions for a given observational dataset.

    Parameters
    ----------
    obsPredict : Dict
        A nested dictionary composed of trend values. The dictionary
        has the form trends[model][member][period]. The climate model and 
        ensemble member are specified with a string and the period is a tuple of
        int years (start, end). The end year is inclusive (i.e., goes
        through December 31). Each entry is a list of trends in the following
        order: total, forced, unforced.

    ds : str
        Observational dataset to retrieve. Values include:
            - GISTEMP
            - BEST
            - HadCRUT4-UAH
            - ERA5
            - HadCRUT5
            - ERA5_masked_$DS
        For the final option ERA5 is masked with the missing data of either
        GISTEMP, BEST, or HadCRUT5 (specified with $DS). 

    period : tuple
        Tuple of the data period to retrieve (start_year, end_year). Note that
        the end_year is inclusive (e.g., goes through December of the end_year).

    models : list, optional
        Each prediction is done with one climate model left out. This optional
        argument specifies the climate models that should be considered when
        retrieving the predictions. Default is None.

    Returns
    -------
    yp : np.array
        Array of observational predictions [n_samples, n_ptypes]

    """
    # if models are not specified, get all values
    if not models:
        models = list(obsPredict.keys())
    # get ensemble members for specified dataset
    if ds in obsPredict[models[0]].keys():
        members = list(obsPredict[models[0]][ds].keys())
    else:
        return None
    # initialize output list
    olist = []
    # loop over all climate models and dataset ensemble members
    for model in models:
        for member in members:
            # append predictions to list
            olist.append(obsPredict[model][ds][member][period])
    # cast to array
    olist = np.array(olist)

    return olist


def get_observed_predictions(obsPredict, period, dsets=None, hadcrut_error=True):
    """
    yp = get_observed_predictions(obsPredict, period, hadcrut_error=True)

    Function fetches observational predictions for a set of observational datasets.

    Parameters
    ----------
    obsPredict : Dict
        A nested dictionary composed of trend values. The dictionary
        has the form trends[model][member][period]. The climate model and 
        ensemble member are specified with a string and the period is a tuple of
        int years (start, end). The end year is inclusive (i.e., goes
        through December 31). Each entry is a list of trends in the following
        order: total, forced, unforced.

    period : tuple, optional
        Tuple of the data period to retrieve (start_year, end_year). Note that
        the end_year is inclusive (e.g., goes through December of the end_year).

    dsets : list, optional
        List of datasets (string values) from which to retrieved predictions. If not
        specified, the function will return all datasets (except for the masked
        variants of ERA5).

    hadcrut_error : Boolean, optional
        Specified whether the HadCRUT5 distribution should be centered on individual
        observational datasets with only one ensemble member. Default is True.

    Returns
    -------
    yp : np.array
        Array of observational predictions [n_samples, n_ptypes]

    """
    # get a list of climate models
    models = list(obsPredict.keys())
    # if the datasets are not specified get all datasets except the masked ERA5 datasets
    if not dsets:
        dsets = [ds for ds in obsPredict[models[0]].keys() if '_masked' not in ds]
        dset = [ds for ds in dsets if period in obsPredict[models[0]][ds].keys()]
    # initialize output array
    all_obs = []
    # loop over climate models
    for model in models:
        # get hadcrut5 distribution for leave-one-out climate model
        if 'HadCRUT5' in obsPredict[model].keys():
            hadobs = get_observed_predictions_for_dataset(obsPredict, 'HadCRUT5', period, models=[model])
            hadkey = 'HadCRUT5'
        elif 'HadSST4' in obsPredict[model].keys():
            hadobs = get_observed_predictions_for_dataset(obsPredict, 'HadSST4', period, models=[model])
            hadkey = 'HadSST4'
        hadobs_centered = hadobs - np.mean(hadobs, axis=0)
        # loop over all datasets
        for ds in dsets:
            # get predictions for dataset
            opreds = get_observed_predictions_for_dataset(obsPredict, ds, period, models=[model])
            # if it is a single dataset and hadcrut5 errror is to be included
            # add the hadcrut5 (zero-centered) distribution to predictions
            if (hadcrut_error & (ds != hadkey) & (opreds is not None)):
                opreds = opreds + hadobs_centered
            # append results to list
            if opreds is not None:
                all_obs = all_obs + list(opreds)
    # cast to array
    all_obs = np.array(all_obs)

    return all_obs


def get_distribution(x, y, x_obs, x_dist, rtype='linear', weights=None, lbound=0.025, ubound=0.975):
    """
    l, u, c, P = get_distribution(x, y, x_obs, x_dist)

    Function creates a linear model between a predictor vector x and predictand vector y. It then
    determines the expected value (including error) given observations x_obs (analagous to an
    emergent constraint).

    Parameters
    ----------
    x : list | array
        Vector of predictor values.

    y : list | array
        Vector of predictand values.

    x_obs : list | array
        List of observed values.

    x_dist : np.array
        Array of x-values for output probability density function.

    rtype : str, optional
        Regression type. Can be:
            - 'linear': linear least-squares regression (default)
            - 'orthogonal': orthogonal distance regression

    weights : list | array, optional
        Vector of weights (default None). Only used in orthoregress.

    lbound : float
        Specify lower bound of confidence interval (e.g., 0.05 for 90% CI).
        Default is 0.025 for 95% CI.

    ubound : float
        Specify upper bound of confidence interval (e.g., 0.95 for 90% CI).
        Default is 0.975 for 95% CI.

    Returns
    -------
    l : float
        Lower bound of expected value.

    u : float
        Upper bound of expected value.

    c : float
        Central estimate of expected value.

    P : np.array
        Probability density function values corresponding to x_dist.

    Notes
    -----
    Based on Methods in:

    Cox, P. M., C. Huntingford, M. S. Williamson, 2018: "Emergent constraint on equilibrium 
    climat sensitivity from global temperature variability," Nature, doi: 10.1038/nature25450.

    Linear regression follows:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html

    Orthogonal distance regression is outlined in fx_odr.py and is derived from code by Robin
    Wilson.

    """
    # ensure input predictor/predictand values are np.arrays
    x = np.array(x)
    y = np.array(y)
    # number of predictors
    N = len(x)
    # least-squares linear fit
    if rtype == 'linear':
        m, b, r, p, e = linregress(x, y)
    elif rtype == 'orthogonal':
        m, b, r, p, e = orthoregress(x, y, weights=weights)
        r = np.corrcoef(x, y)[0, 1]
    # estimated y for given x
    f = m*x+b
    # squared error
    s2 = 1./(N-2)*np.sum(np.power(y - f, 2))
    # rms
    s = np.sqrt(s2)
    sx = np.std(x)
    sx2 = np.power(sx, 2)
    # pre-allocate probability distribution
    P = np.zeros(len(x_dist))
    # loop over all obs values of x and sum up probabilities
    for xi in x_obs:
        # prediction error for x
        sf = s * np.sqrt(1 + 1./N + np.power(xi - np.mean(x), 2)/(N * sx2))
        # probability density
        sf2 = np.power(sf, 2)
        fi = m*xi + b
        p = 1./np.sqrt(2*np.pi*sf2) * np.exp(-np.power(x_dist - fi, 2)/(2*sf2))
        # sum probabilities over all obs
        P = P + p
    # re-normalize distribution
    P = P/len(x_obs)
    # calculate cumulative probability
    cdf = np.zeros(len(P)-1)
    for i in range(len(P)-1):
        p = np.trapz(P[0:i+1], x_dist[0:i+1])
        cdf[i] = p
    # get 95% CI
    l = np.interp(lbound, cdf, x_dist[1:])
    u = np.interp(ubound, cdf, x_dist[1:])
    c = np.interp(0.5, cdf, x_dist[1:])

    return l, u, c, P


def get_bias_corrected_prediction(trends, trendPredict, obsPredict, period, x_dist=np.arange(-0.5, 1.5, 0.001)):
    """
    bias_corrected_prediction = get_bias_corrected_prediction(trends, trendPredict, obsPredict, period)

    Function creates a linear model between a predictor vector x and predictand vector y. It then
    determines the expected value (including error) given observations x_obs (analagous to an
    emergent constraint).

    Parameters
    ----------
    trends : Dict
        A nested dictionary composed of trend values. The dictionary
        has the form trends[model][member][period]. The climate model and 
        ensemble member are specified with a string and the period is a tuple of
        int years (start, end). The end year is inclusive (i.e., goes
        through December 31). Each entry is a list of trends in the following
        order: total, forced, unforced.

    trendPredict : dict
        A nested dictionary composed of trend values. The dictionary
        has the form trends[model][member][period]. The climate model and 
        ensemble member are specified with a string and the period is a tuple of
        int years (start, end). The end year is inclusive (i.e., goes
        through December 31). Each entry is a list of trends stored in the 
        following order: total, forced, unforced.

    obsPredict : Dict
        A nested dictionary composed of trend values. The dictionary
        has the form trends[model][member][period]. The climate model and 
        ensemble member are specified with a string and the period is a tuple of
        int years (start, end). The end year is inclusive (i.e., goes
        through December 31). Each entry is a list of trends in the following
        order: total, forced, unforced.

    period : tuple, optional
        Tuple of the data period to retrieve (start_year, end_year). Note that
        the end_year is inclusive (e.g., goes through December of the end_year).

    x_dist : np.array, optional
        Array of x-values for output probability density function. Default is:
        np.arange(-0.5, 1.5, 0.001).

    Returns
    -------
    bias_corrected_prediction : dict
        Bias corrected prediction with the following keys:
            - 'prediction' (float): expected value based on observations
            - 'upper' (float): upper bound for prediction
            - 'lower' (float): lower bound for prediction
            - 'error' (float):      95% CI interval estimate
            - 'p' (np.array):       probability density (corresponding to x_dist)
    """
    # initialize vectors for regression
    xall = []
    yall = []
    wall = []
    models = list(trends.keys())
    for model in models:
        x, y = get_predicted_and_actual_trends(trends, trendPredict, model, period=period, leave_out_periods=[], ptype=['total', 'forced', 'unforced'])
        xall = xall + list(x)
        yall = yall + list(y)
        w = np.ones(len(x)) * 1 / len(x)
        wall = wall + list(w)
    xall = np.array(xall)
    yall = np.array(yall)
    x_obs = get_observed_predictions(obsPredict, period)
    # hardcode trend dictionary matrix
    trendDictMapping = {'total': 0, 'forced': 1, 'unforced': 2}
    bcp = {'total': {}, 'forced': {}, 'unforced': {}}
    for ptype in ['total', 'forced', 'unforced']:
        pind = trendDictMapping[ptype]
        l, u, c, p = get_distribution(xall[:, pind], yall[:, pind], x_obs[:, pind], x_dist, rtype='orthogonal', weights=wall)
        e = (u - l) / 2
        # ensure +/- error estimate is close to CDF values on both sides
        if np.abs((u-c) - (c-l)) > 0.005:
            ValueError('error estimate is not sufficiently close')
        bcp[ptype] = {'prediction': c, 'error': e, 'distribution': p, 'upper': u, 'lower': l}
    return bcp


def perform_leave_one_out_fit(model_type, trends, trendMaps, obsTrendMaps, period, n_members=10, ml_args={}, weight=None, infill='zonal'):
    """
    trendPredict, obsPredict, MSE, ml_coefs, bcp = perform_leave_one_out_fit(model_type, trends, trendMaps, obsTrendMaps, n_members, period)

    Function creates a linear model between a predictor vector x and predictand vector y. It then
    determines the expected value (including error) given observations x_obs (analagous to an
    emergent constraint).

    Parameters
    ----------
    model_type : str
        Model type to use. Can specify:
            - 'nn': MLPRegressor neural network
            - 'ridge': Ridge regression
            - 'pls': PLS regression

    trends : Dict
        A nested dictionary composed of trend values. The dictionary
        has the form trends[model][member][period]. The climate model and 
        ensemble member are specified with a string and the period is a tuple of
        int years (start, end). The end year is inclusive (i.e., goes
        through December 31). Each entry is a list of trends in the following
        order: total, forced, unforced.

    trendMaps : Dict
        A nested dictionary composted of trend maps. The dictionary
        has the form trendMaps[model][member][period] (as with trends).

    obsTrendMaps : Dict
        A nested dictionary composed of trend values. The dictionary
        has the form trends[model][member][period]. The climate model and 
        ensemble member are specified with a string and the period is a tuple of
        int years (start, end). The end year is inclusive (i.e., goes
        through December 31). Each entry is a list of trends in the following
        order: total, forced, unforced.

    period : tuple, optional
        Tuple of the data period to evaluate (start_year, end_year). Note that
        the end_year is inclusive (e.g., goes through December of the end_year).
        This period is left out of training.

    n_members : int, optional
        Number of members to be considered from each climate model for training.
        Default is 10.

    ml_args : dict, optional
        Parameters specified in producing mlmodel. See initialize_ml_model for
        more details.

    weight : np.array, optional
        Matrix of weights (e.g., spatial) that can be multiplied with the
        predictor matrix (if included, the default is None). If weights values
        are zero, these regions will be removed from the predictor matrix.

    Returns
    -------

    trendPredict : dict
        A nested dictionary composed of trend values. The dictionary
        has the form trends[model][member][period]. The climate model and 
        ensemble member are specified with a string and the period is a tuple of
        int years (start, end). The end year is inclusive (i.e., goes
        through December 31). Each entry is a list of trends stored in the 
        following order: total, forced, unforced.

    obsPredict : dict
        A nested dictionary composed of trend values. The dictionary
        has the form trends[model][member][period]. The climate model and 
        ensemble member are specified with a string and the period is a tuple of
        int years (start, end). The end year is inclusive (i.e., goes
        through December 31). Each entry is a list of trends in the following
        order: total, forced, unforced.

    MSE : list
        Mean squared error for the predicted-versus-actual trend values over
        all time periods except for the period specified. The number of rows
        corresponds to the number of leave-one-out iterations and each row
        contains the MSE for the total, forced, and unforced trend.

    ml_coefs : dict
        A nested dictionary composed of the fit coefficients in the form of
        ml_coefs[coefficient][model] where the coefficient and climate model
        are both strings. The coefficient values vary by scikit-learn model
        type. A neural network ('nn') has no coefficients. See store_coefficients
        for more information.

    bias_corrected_prediction : dict
        Bias corrected prediction with the following keys:
            - 'prediction' (float): expected value based on observations
            - 'error' (float):      95% CI interval
            - 'p' (np.array):       probability density (corresponding to x_dist)
    """
    # initialize output dictionaries / lists
    trendPredict = {}
    obsPredict = {}
    ml_coefs = {}
    MSE = []
    # get models (to be used in leave-one-out loop)
    models = list(trends.keys())
    # get record_len based on specified period
    record_len = period[1] - period[0] + 1
    # adjust weights
    maskData = False
    if len(np.where(weight == 0)[0]) > 0:
        data_inds = np.where(np.squeeze(weight) != 0)[0]
        weight = weight[:, data_inds]
        maskData = True
    else:
        data_inds = None
    # loop over models
    for model in models:
        # initialize ML model
        mlmodel, model_args = initialize_ml_model(model_type, **ml_args)
        # get training data; standardize data; fit model; store coefficient
        XT, YT = get_training_matrix(trends, trendMaps, leave_out_models=[model],
                                     leave_out_periods=[period],
                                     record_len=record_len,
                                     n_members=n_members,
                                     ptype=['unforced', 'forced'])
        if maskData:
            XT = XT[:, data_inds]
        XT, YT, XM, XS, YM, YS = standardize(XT, YT, weight=weight)
        mlmodel.fit(XT, YT)
        ml_coefs = store_coefficients(ml_coefs, model, model_type, mlmodel, (72, 144), data_inds=data_inds)
        # do leave-one-out prediction
        XP, YA, fit_labels = get_model_data(trends, trendMaps, model, ptype=['unforced', 'forced'])
        if maskData:
            XP = XP[:, data_inds]
        XP = apply_predictor_scaling(XP, XM, XS, weight=weight)
        yp = mlmodel.predict(XP)
        yp = unscale_predictands(yp, YM, YS)
        trendPredict = predictions_to_dict(trendPredict, yp, fit_labels, ptype=['unforced', 'forced'])
        # do observation-based prediction
        XO, obs_labels = get_obs_predictor_data(obsTrendMaps, period, infill='zonal')
        if maskData:
            XO = XO[:, data_inds]
        XO = apply_predictor_scaling(XO, XM, XS, weight=weight)
        ypo = mlmodel.predict(XO)
        ypo = unscale_predictands(ypo, YM, YS)
        obsPredict = observed_predictions_to_dict(obsPredict, ypo, model, obs_labels, ptype=['unforced', 'forced'])
        # compute overall MSE for model data
        x, y = get_predicted_and_actual_trends(trends, trendPredict, model, period=None, leave_out_periods=[period], ptype=['total', 'forced', 'unforced'])
        mse = [mean_squared_error(y[:, i], x[:, i]) for i in range(3)]
        MSE.append(mse)
    # compute corrected prediction
    bcp = get_bias_corrected_prediction(trends, trendPredict, obsPredict, period, x_dist=np.arange(-0.5, 1.5, 0.001))

    return trendPredict, obsPredict, MSE, ml_coefs, bcp


def weighted_cov(x, y, w):
    """Weighted Covariance"""
    return np.sum(w * (x - np.average(x, weights=w)) * (y - np.average(y, weights=w))) / np.sum(w)


def weighted_corrcoef(x, y, w):
    """
    r = weighted_corrcoef(x, y, w)

    Calculate the weighted correlation between two vectors
    for a given weight at each point.

    Parameters
    ----------

    x : array_like
        A 1-D array of observations.

    y : array_like
        A 1-D array of observations. `y` has the same length as `x`.

    w : array_like
        A 1-D array of weights. `w` has the same length as `x`.

    Returns
    -------

    r : float
        Weighted correlation coefficient.

    """
    x = np.array(x)
    y = np.array(y)
    w = np.array(w)
    return weighted_cov(x, y, w) / np.sqrt(weighted_cov(x, x, w) * weighted_cov(y, y, w))


def make_scatter_plot(trends, trendPredict, obsPredict, marker_map, color_map, obsColors, period, ptype, models, plot_options):
    """
    make_scatter_plot(trends, trendPredict, obsPredict, marker_map, color_map, obsColors, period, ptype, models, plot_options)

    Create standard scatter plot displayed in this project.

    Parameters
    ----------

    trends : Dict
        A nested dictionary composed of trend values. The dictionary
        has the form trends[model][member][period]. The climate model and
        ensemble member are specified with a string and the period is a tuple of
        int years (start, end). The end year is inclusive (i.e., goes
        through December 31). Each entry is a list of trends in the following
        order: total, forced, unforced.

    trendPredict : dict
        A nested dictionary composed of trend values. The dictionary
        has the form trends[model][member][period]. The climate model and
        ensemble member are specified with a string and the period is a tuple of
        int years (start, end). The end year is inclusive (i.e., goes
        through December 31). Each entry is a list of trends stored in the
        following order: total, forced, unforced.

    obsPredict : dict
        A nested dictionary composed of trend values. The dictionary
        has the form trends[model][member][period]. The climate model and
        ensemble member are specified with a string and the period is a tuple of
        int years (start, end). The end year is inclusive (i.e., goes
        through December 31). Each entry is a list of trends in the following
        order: total, forced, unforced.

    marker_map : dict
        A dictionary mapping each climate model (the key) to a specified marker
        style.

    color_map : dict
        A dictionary mapping each climate model (the key) to a specified color.

    obsColors : dict
        A dictionary mapping each observational dataset (the key) to a specified
        color.

    period : tuple, optional
        Tuple of the data period to evaluate (start_year, end_year). Note that
        the end_year is inclusive (e.g., goes through December of the end_year).
        This period is left out of training.

    ptype : str
        The type of trend prediction to be considered (either 'forced',
        'unforced', and 'total').

    models : list
        List of climate models to be included in scatter plot.

    plot_options : dict
        Dictionary of plot specifications. These include the following
        key-value pairs:

        Key                 |   value
        ----------------------------------------------------------------------
        legend              |   dict: legend options (see below)
        limits              |   dict: limits options (see below)
        xlabel              |   str: x-axis label
        ylabel              |   str: x-axis label
        smarker             |   bool: option to include scatter markers
        rline               |   bool: option to include model regression lines
        title               |   str: plot title
        xticks              |   array: x-major tick values
        yticks              |   array: y-major tick values
        spine_offset        |   list: distance to offset spines [x, y]
        obs_y_location      |   list: vertical placement and spacing of 
                                      observations [y, dy]
        obs_xl_location     |   float: observations label x-value
        obslabels           |   bool: option to include observation labels
        pdfheight           |   float: max height of PDF
        rlabel              |   list: [x, y] location of correlation label
        vlabel              |   list: [x, y] location of prediction label
        obstrends           |   dic: MSU values dict[dataset][period] (or None)
        xobs_shade          |   bool: option to shade observed range (in x-dimension)

    Note that legend and limits are separate dictionaries that are specified
    such that limits include a key of `x` and `y` with a list of upper and 
    lower limit values. The legend has the following key / value pairs:

        Key                 |   Value
        ----------------------------------------------------------------------
        on                  |   bool, option to plot model legend
        x                   |   float, x-location of legend
        y                   |   float, y-location of legend
        vs                  |   float, vertical spacing of legend entries
        hs                  |   float, horizontal space between legend marker/text
        th                  |   float, text height

    """
    xall = []
    yall = []
    wall = []
    # hardcode trend dictionary matrix
    trendDictMapping = {'total': 0, 'forced': 1, 'unforced': 2}
    # loop over and plot each model
    for i, model in enumerate(models):
        # get plot options for each model
        ms = marker_map[model]
        c = color_map[model]
        # get predicted and actual trends
        x, y = get_predicted_and_actual_trends(trends, trendPredict, model, period=period, leave_out_periods=[], ptype=[ptype])
        # make 1d vectors
        x = x[:, 0]; y = y[:, 0]
        # append model data to all-model-list
        xall = xall + list(x)
        yall = yall + list(y)
        wall = wall + list(np.ones(len(x)) / len(x))
        # do individual model scatter (with marker option)
        if plot_options['smarker']:
            plt.plot(x, y, ms, color=c, markersize=2)
        else:
            plt.plot([np.min(x), np.max(x)], [np.min(y), np.max(y)], color=c)
            plt.plot(np.mean(x), np.mean(y), ms, color=c, markersize=2)
        # plot best-fit line
        if plot_options['rline']:
            m, b, r, p, e = orthoregress(x, y, weights=None)
            x1 = np.arange(np.min(x), np.max(x), 0.001)
            plt.plot(x1, x1*m+b, color=c, linewidth=1)
        # legend (if specified)
        lx = plot_options['legend']['x']
        ly = plot_options['legend']['y']
        lvs = plot_options['legend']['vs']
        lhs = plot_options['legend']['hs']
        lth = plot_options['legend']['th']
        if plot_options['legend']['on']:
            r = np.corrcoef(x, y)[0, 1]
            rs = '{:.2f}'.format(np.round(r, 2))
            mtext = model + ' (' + str(len(x)) + ' / ' + rs + ')'
            plt.plot([lx - lhs], [ly -i*lvs +lth], ms, color=c, markersize=3)
            plt.text(lx, ly-i*lvs, mtext, color=c)
    # plot observations
    if 'obs_y_location' in plot_options.keys():
        oyl = plot_options['obs_y_location'][0]
        oys = plot_options['obs_y_location'][1]
        dscount = 0
        xoall = []
        for i, dset in enumerate(obsColors.keys()):
            c = obsColors[dset]
            xo = get_observed_predictions(obsPredict, period, dsets=[dset])
            if len(xo) == 0:
                continue
            else:
                dscount += 1
            index = trendDictMapping[ptype]
            xo = xo[:, index]
            xoall = xoall + list(xo)
            plt.plot([np.min(xo), np.max(xo)], np.array([oyl, oyl])-dscount*oys, color=c, solid_capstyle='butt')
            if plot_options['obslabels']:
                plt.text(np.max(xo) + lhs, oyl-dscount*oys-lth, dset, color=c)
        if plot_options['xobs_shade']:
            plt.axvspan(np.min(xoall), np.max(xoall), color='gray', alpha=0.25, linewidth=0)
    # add legend labels
    if plot_options['legend']['on']:
        plt.text(lx, ly+lvs, 'Model (n / r)', color='k', fontweight='bold')
    if 'obs_xl_location' in plot_options.keys():
        oxl = plot_options['obs_xl_location']
        plt.text(oxl, oyl, 'Observations', color='k', fontweight='bold')
    # get ticks (to help place PDF)
    yticks = plot_options['yticks']
    xticks = plot_options['xticks']
    # plot distribution
    x_dist = np.arange(np.min(yticks), np.max(yticks), 0.001)
    bcp = get_bias_corrected_prediction(trends, trendPredict, obsPredict, period, x_dist=x_dist)
    c = bcp[ptype]['prediction']
    eu = bcp[ptype]['upper']
    el = bcp[ptype]['lower']
    e = bcp[ptype]['error']
    P = bcp[ptype]['distribution']
    if 'pdfheight' in plot_options.keys():
        PN = P / np.max(P) * plot_options['pdfheight']
        # get point values
        cx = np.interp(c, x_dist, PN)
        exu = np.interp(eu, x_dist, PN)
        exl = np.interp(el, x_dist, PN)
        # plot distribution and markers
        plt.plot(PN + xticks[0]+0.005, x_dist, 'k')
        plt.plot([exl+xticks[0]+0.005, exu+xticks[0]+0.005], [el, eu], 'k_')
        plt.plot(cx+xticks[0]+0.005, c, 'ko', markersize=4)
    # overall fit
    m, b, _, _, _ = orthoregress(xall, yall, weights=wall)
    x1 = np.arange(np.min(xall), np.max(xall), 0.001)
    plt.plot(x1, x1*m+b, linewidth=2.5, color='gray', alpha=0.6)
    # overall correlation coefficient
    if 'rlabel' in plot_options.keys():
        rxl = plot_options['rlabel'][0]
        ryl = plot_options['rlabel'][1]
        r = weighted_corrcoef(xall, yall, wall)
        rs = '{:.2f}'.format(np.round(r, 2))
        plt.text(rxl, ryl, 'r = ' + rs)
    # overall prediction
    if 'vlabel' in plot_options.keys():
        vxl = plot_options['vlabel'][0]
        vyl = plot_options['vlabel'][1]
        cs = '{:.2f}'.format(np.round(c, 2))
        es = '{:.2f}'.format(np.round(e, 2))
        plt.text(vxl, vyl, cs + ' $\pm$ ' + es)
    # plot observed trends
    if plot_options['obstrends'] is not None:
        obsTrends = plot_options['obstrends']
        olist = [obsTrends[ds][period] for ds in obsTrends.keys()]
        plt.axhspan(np.min(olist), np.max(olist), color='purple', alpha=0.25, linewidth=0)
    # one-to-one line
    plt.plot(plot_options['limits']['y'], plot_options['limits']['y'], 'k--', linewidth=1.5)
    # ticks
    ax = plt.gca()
    yts = yticks[1] - yticks[0]
    xts = xticks[1] - xticks[0]
    yticksm = (np.arange(yticks[0] + yts/2, yticks[-1], yts))
    xticksm = (np.arange(yticks[0] + yts/2, yticks[-1], xts))
    ax.set_xticks(xticks)
    ax.set_xticks(xticksm, minor=True)
    ax.set_yticks(yticks)
    ax.set_yticks(yticksm, minor=True)
    # spines
    ax.tick_params(axis='both', which='major')
    ax.tick_params(axis='both', which='minor')
    sox = plot_options['spine_offset'][0]
    soy = plot_options['spine_offset'][1]
    ax.spines.left.set_position(('data', np.min(xticks)+sox))
    ax.spines.left.set_bounds((np.min(yticks), np.max(yticks)))
    ax.spines.right.set_color('none')
    ax.spines.bottom.set_position(('data', plot_options['limits']['y'][0]+soy))
    ax.spines.bottom.set_bounds((np.min(xticks), np.max(xticks)))
    ax.spines.top.set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    # plot limits
    plt.xlim(plot_options['limits']['x'])
    plt.ylim(plot_options['limits']['y'])
    # plot labels
    plt.xlabel(plot_options['xlabel'])
    plt.ylabel(plot_options['ylabel'], loc='top')
    plt.title(plot_options['title'], fontsize=10)
    plt.tight_layout()


def make_map(ax, bmap, lat, lon, clevs, title, cmap=plt.cm.RdBu_r, extend='both'):
    """
    im = make_map(ax, bmap, lat, lon, clevs, title)

    Create standard map displayed in this project.

    Parameters
    ----------
    ax : matplotlib axis
        Axis to plot map.

    bmap : np.array
        Map (lat, lon) of data to contour.

    lat : np.array
        latitude values

    lon : np.array
        longitude values

    clevs : np.array
        contour intervals for plot

    title : str
        title for plot

    cmap : matplotlib color map, optional
        colormap for plot

    extend : str, optional
        option to extend colormap (from matplotlib)

    Returns
    -------

    im : cartopy contour object
        color contour returned from map

    """
    plt.title(title, loc='left', fontsize=10)
    bmap, clon = add_cyclic_point(bmap, coord=lon)
    im = plt.contourf(clon, lat, bmap, clevs, cmap=cmap, transform=ccrs.PlateCarree(), extend=extend)
    ax.set_global()
    ax.coastlines()
    return im
