#!/bin/env python
# -*- coding: utf-8 -*-

"""
Stephen Po-Chedley 25 August 2022

Create figure 2: fingerprints and warming map.

Copyright 2022, Stephen Po-Chedley, All rights reserved.

@author: pochedls
"""

# %% imports
import pickle
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import sys
sys.path.append('..')
from fx import get_lat_lon_weight, weighted_corrcoef, get_obs_predictor_data, make_map


# %% Parameters
domain = 'tropical'
period = (1979, 2014)
obsdatasets = ['HadCRUT4-UAH', 'ERA5', 'GISTEMP', 'BEST', 'HadCRUT5']

# %% specify / construct filenames
fn_obs = '../data/trendmaps/obs_tas_trendmaps.pickle'
fn_mlmodel = '../data/mlmodels/plsModel_tropical_1979-2014.pickle'

# %% load data
obsTrends = pickle.load(open(fn_obs, 'rb'))
mlmodel = pickle.load(open(fn_mlmodel, 'rb'))
lat, lon, weight = get_lat_lon_weight()
weight = np.squeeze(weight)  # remove singleton

# %% Set global font size
font = {'weight': 'normal',
        'size': 9}
plt.rc('font', **font)

# %% get beta maps
models = list(mlmodel['beta'])
# get array of beta maps
bmaps = np.array([mlmodel['beta'][model] for model in mlmodel['beta'].keys()])
# calculate multimodel mean
mmbmap = np.mean(bmaps, axis=0)

# %% check spatial correlation between individual maps and multimodel mean
# loop over unforced/forced then models
for i, ptype in enumerate(['unforced', 'forced']):
    rvec = []
    refmap = np.reshape(mmbmap[i, :, :], -1)
    for im, model in enumerate(models):
        mmap = np.reshape(bmaps[im, i, :, :], -1)
        r = weighted_corrcoef(refmap, mmap, weight)
        rvec.append(r)
    print('Minimum r-value for ' + ptype + ' maps is: ' + str(np.min(rvec)))

# %% get observational mean
omaps = []
for ds in obsdatasets:
    mmap, _ = get_obs_predictor_data(obsTrends, period, infill='zonal', datasets=[ds])
    mmap = np.mean(mmap, axis=0)
    mmap = np.reshape(mmap, (len(lat), len(lon)))
    omaps.append(mmap)
omaps = np.mean(np.array(omaps), axis=0)

# %% make plot
# define maps to plot
plt.figure(figsize=(3.5, 6.5), dpi=300)
pmaps = [mmbmap[0, :, :]*250., mmbmap[1, :, :]*250., omaps]
clevs = np.arange(-0.6, 0.61, 0.1)
ptitles = ['a. Unforced Fingerprint', 'b. Forced Fingerprint', 'c. Observed Warming']
for i in range(3):
    ax = plt.subplot(3, 1, i+1, projection=ccrs.Robinson(central_longitude=180.))
    im = make_map(ax, pmaps[i], lat, lon, clevs, ptitles[i])
# colorbar
fig = plt.gcf()
cbar_ax = fig.add_axes([0.1, 0.075, 0.8, 0.025])
cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
plt.subplots_adjust(wspace=0.025, hspace=0.005)
plt.savefig('../figures/f2_fingerprints.png', bbox_inches='tight')
plt.savefig('../figures/f2_fingerprints.pdf', bbox_inches='tight')
plt.show()




