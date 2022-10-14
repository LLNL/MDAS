#!/bin/env python
# -*- coding: utf-8 -*-

"""
Stephen Po-Chedley 23 August 2022

Code to do leave-one-out training / predictions of the components
of tropospheric warming.

This was run using:
    - Main result
        period = (1979, 2014)
        n_members = 10
    - Extended through 2021
        period = (1979, 2021)
        n_members = 10

Yields the following dictionaries:
    trendPredict : dictionary of predicted trends (for models)
        [output from fx.perform_leave_one_out_fit]
    obsPredict : dictionary of predicted trends (for observations)
        [output from fx.perform_leave_one_out_fit]
    ml_coefs : ML model coefficients
        [output from fx.store_coefficients]
    bcp : dictionary of bias-corrected predictions
        [output includes the prediction, error, distribution, and
         upper/lower bounds]

Results are written to data/predictions/ and data/mlmodels/.

@author: pochedls
"""

# %% imports
import pickle
import numpy as np
from fx import perform_leave_one_out_fit, get_lat_lon_weight

# %% parameters
n_members = 10
period = (1979, 2014)
pstring = str(period[0]) + '-' + str(period[1])
domains = ['tropical', 'global']
ml_models = ['pls', 'ridge', 'nn']
lat, lon, weight = get_lat_lon_weight()
weighting = {'pls': weight, 'ridge': None, 'nn': None}
mlparameters = {'nn': {'alpha': 30, 'verbose': False},
                'pls': {'n_components': 6},
                'ridge': {'alpha': 17500}}
# load data based on time period
if period == (1979, 2014):
    fn_msu_trends = 'data/trends/cmip6_ttt_trends.pickle'
    fn_tas_trends = 'data/trendmaps/cmip6_tas_trendmaps.pickle'
    fn_obs_tas_trends = 'data/trendmaps/obs_tas_trendmaps.pickle'
elif period == (1979, 2021):
    fn_msu_trends = 'data/trends/cmip6_ttt_spliced_trends.pickle'
    fn_tas_trends = 'data/trendmaps/cmip6_tas_spliced_trendmaps.pickle'
    fn_obs_tas_trends = 'data/trendmaps/obs_tas_trendmaps.pickle'
else:
    ValueError('Time period not supported')


# %% load data
allTrends = pickle.load(open(fn_msu_trends, 'rb'))
trendMaps = pickle.load(open(fn_tas_trends, 'rb'))
obsTrendMaps = pickle.load(open(fn_obs_tas_trends, 'rb'))

# %% loop over domains and ml models and do predictions
for domain in domains:
    trends = allTrends[domain]
    for mlmodel in ml_models:
        weight = weighting[mlmodel]
        ml_args = mlparameters[mlmodel]
        trendPredict, obsPredict, mse, ml_coefs, bcp = perform_leave_one_out_fit(mlmodel, trends, trendMaps, obsTrendMaps, period, n_members=n_members, ml_args=ml_args, weight=weight)
        print(domain + ' / ' + mlmodel)
        for ptype in ['total', 'forced', 'unforced']:
            s = str(np.round(bcp[ptype]['prediction'], 3)) + ' ± ' + str(np.round(bcp[ptype]['error'], 3))
            print(ptype + ' ' + s)
        print()
        pickle.dump(trendPredict, open('data/predictions/trendPredict_' + mlmodel + '_' + domain + '_' + pstring + '.pickle', 'wb'))
        pickle.dump(obsPredict, open('data/predictions/obsPredict_' + mlmodel + '_' + domain + '_' + pstring + '.pickle', 'wb'))
        pickle.dump(ml_coefs, open('data/mlmodels/' + mlmodel + 'Model_' + domain + '_' + pstring + '.pickle', 'wb'))
        pickle.dump(bcp, open('data/predictions/overallPrediction_' + mlmodel + '_' + domain + '_' + pstring + '.pickle', 'wb'))

# %% do a one-off prediction with SST
# sst parameters
mlmodel = 'pls'
domain = 'tropical'
period = (1979, 2014)
n_members = 10
ml_args = mlparameters[mlmodel]

# specify sst data
fn_tas_trends = 'data/trendmaps/cmip6_ts_trendmaps.pickle'
fn_obs_tas_trends = 'data/trendmaps/obs_sst_trendmaps.pickle'
fn_msu_trends = 'data/trends/cmip6_ttt_trends.pickle'

# %% load data
trendMaps = pickle.load(open(fn_tas_trends, 'rb'))
obsTrendMaps = pickle.load(open(fn_obs_tas_trends, 'rb'))
rmap = obsTrendMaps['HadSST4'][1][period]
allTrends = pickle.load(open(fn_msu_trends, 'rb'))
trends = allTrends[domain]

# %% Adjust weights to zero out extratropics and land
lat, lon, weight = get_lat_lon_weight()
weight = np.reshape(weight, (len(lat), len(lon)))
# zero outside of 40N-S
inds = np.where(np.abs(lat) > 40)[0]
weight[inds, :] = 0.
weight = np.where(np.isnan(rmap), 0., weight)
weight = np.expand_dims(np.reshape(weight, -1), axis=0)

trendPredict, obsPredict, mse, ml_coefs, bcp = perform_leave_one_out_fit(mlmodel, trends, trendMaps, obsTrendMaps, period, n_members=n_members, ml_args=ml_args, weight=weight)
print('SST Prediction: ' + domain + ' / ' + mlmodel)
for ptype in ['total', 'forced', 'unforced']:
    s = str(np.round(bcp[ptype]['prediction'], 3)) + ' ± ' + str(np.round(bcp[ptype]['error'], 3))
    print(ptype + ' ' + s)

pickle.dump(trendPredict, open('data/predictions/trendPredict_' + mlmodel + '_' + domain + '_sst_' + pstring + '.pickle', 'wb'))
pickle.dump(obsPredict, open('data/predictions/obsPredict_' + mlmodel + '_' + domain + '_sst_' + pstring + '.pickle', 'wb'))
pickle.dump(ml_coefs, open('data/mlmodels/' + mlmodel + 'Model_' + domain + '_' + pstring + '_sst.pickle', 'wb'))
pickle.dump(bcp, open('data/predictions/overallPrediction_' + mlmodel + '_' + domain + '_' + pstring + '_sst.pickle', 'wb'))
