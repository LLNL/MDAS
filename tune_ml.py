#!/bin/env python
# -*- coding: utf-8 -*-

"""
Stephen Po-Chedley 22 August 2022

Tune ML parameters.

Results are written to data/tuning/.

Each dictionary includes the MSE, prediction, and error as a
function of parameter value.

@author: pochedls
"""

# %% imports
import pickle
import numpy as np
from fx import perform_leave_one_out_fit, get_lat_lon_weight
import time

# %% parameters
fn_msu_trends = 'data/trends/cmip6_ttt_trends.pickle'
fn_tas_trends = 'data/trendmaps/cmip6_tas_trendmaps.pickle'
fn_obs_tas_trends = 'data/trendmaps/obs_tas_trendmaps.pickle'
domain = 'tropical'
n_members = 10
period = (1979, 2014)

# %% load data
print()
print('Load data')
print(time.ctime())
trends = pickle.load(open(fn_msu_trends, 'rb'))[domain]
trendMaps = pickle.load(open(fn_tas_trends, 'rb'))
obsTrendMaps = pickle.load(open(fn_obs_tas_trends, 'rb'))
lat, lon, weight = get_lat_lon_weight()

# %% Tune PLS regression
print()
print('Tune PLS Regression')
print(time.ctime())

# PLS parameters
model_type = 'pls'

# initialize tuning dictionaries
MSE = {}
BETA = {}
PREDS = {}
ERROR = {}
# loop over pls tunable parameters
for n_comps in np.arange(2, 11):
    ml_args = {'n_components': n_comps}
    # do leave-one-out to get overall MSE
    trendPredict, obsPredict, mse, ml_coefs, bcp = perform_leave_one_out_fit(model_type, trends, trendMaps, obsTrendMaps, period, n_members=n_members, ml_args=ml_args, weight=weight)
    # store results
    MSE[n_comps] = mse
    BETA[n_comps] = ml_coefs['beta']
    PREDS[n_comps] = [bcp['total']['prediction'], bcp['forced']['prediction'], bcp['unforced']['prediction']]
    ERROR[n_comps] = [bcp['total']['error'], bcp['forced']['error'], bcp['unforced']['error']]
    print(n_comps, np.mean(mse))

# save output
tuningDict = {'BETA': BETA, 'MSE': MSE, 'PREDS': PREDS, 'ERROR': ERROR}
pickle.dump(tuningDict, open('data/tuning/pls.pickle', 'wb'))


# %% Tune ridge regression
print()
print('Tune Ridge Regression')
print(time.ctime())

# ridge parameters
model_type = 'ridge'
weight = None
alphaList = [1, 2, 4, 8, 12, 16, 24, 32, 64, 128, 256, 512, 1000, 2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000, 25000, 30000, 50000, 75000, 100000]

# initialize tuning dictionaries
MSE = {}
BETA = {}
PREDS = {}
ERROR = {}
# loop over ridge tunable parameters
for alpha in alphaList:
    ml_args = {'alpha': alpha}
    # do leave-one-out to get overall MSE
    trendPredict, obsPredict, mse, ml_coefs, bcp = perform_leave_one_out_fit(model_type, trends, trendMaps, obsTrendMaps, period, n_members=n_members, ml_args=ml_args, weight=weight)
    # store results
    MSE[alpha] = mse
    BETA[alpha] = ml_coefs['beta']
    PREDS[alpha] = [bcp['total']['prediction'], bcp['forced']['prediction'], bcp['unforced']['prediction']]
    ERROR[alpha] = [bcp['total']['error'], bcp['forced']['error'], bcp['unforced']['error']]
    print(alpha, np.mean(mse))

# save output
tuningDict = {'BETA': BETA, 'MSE': MSE, 'PREDS': PREDS, 'ERROR': ERROR}
pickle.dump(tuningDict, open('data/tuning/ridge.pickle', 'wb'))

# %% Tune NN
print()
print('Tune Neural Network')
print(time.ctime())

# NN parameters
model_type = 'nn'
alphaList = [1, 2, 4, 6, 8, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300]
weight = None

# initialize tuning dictionaries
MSE = {}
BETA = {}
PREDS = {}
ERROR = {}
# loop over nn tunable parameters
for alpha in alphaList:
    ml_args = {'alpha': alpha, 'verbose': False}
    # do leave-one-out to get overall MSE
    trendPredict, obsPredict, mse, ml_coefs, bcp = perform_leave_one_out_fit(model_type, trends, trendMaps, obsTrendMaps, period, n_members=n_members, ml_args=ml_args, weight=weight)
    # store results
    MSE[alpha] = mse
    PREDS[alpha] = [bcp['total']['prediction'], bcp['forced']['prediction'], bcp['unforced']['prediction']]
    ERROR[alpha] = [bcp['total']['error'], bcp['forced']['error'], bcp['unforced']['error']]
    print(alpha, np.mean(mse))

# save output
tuningDict = {'BETA': BETA, 'MSE': MSE, 'PREDS': PREDS, 'ERROR': ERROR}
pickle.dump(tuningDict, open('data/tuning/nn.pickle', 'wb'))

print()
print('Done')
print(time.ctime())
