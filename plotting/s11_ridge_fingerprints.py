#!/bin/env python
# -*- coding: utf-8 -*-

"""
Stephen Po-Chedley 25 August 2022

Create figure S9: ridge fingerprints

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
from fx import get_lat_lon_weight, weighted_corrcoef, make_map


# %% Parameters
domain = 'tropical'
mlmethod = 'ridge'

# %% specify filenames
fn_mlmodel = '../data/mlmodels/ridgeModel_tropical_1979-2014.pickle'

# %% load data
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

# %% make plot
# define maps to plot
plt.figure(figsize=(8, 4), dpi=300)
pmaps = [mmbmap[0, :, :]*1000, mmbmap[1, :, :]*1000]
clevs = np.arange(-2.5, 2.51, 0.25)
ptitles = ['a. Unforced Fingerprint', 'b. Forced Fingerprint']
for i in range(2):
    ax = plt.subplot(1, 2, i+1, projection=ccrs.Robinson(central_longitude=180.))
    im = make_map(ax, pmaps[i], lat, lon, clevs, ptitles[i])
# colorbar
fig = plt.gcf()
cbar_ax = fig.add_axes([0.92, 0.3, 0.025, 0.4])
cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical')
fig.text(0.92, 0.25, 'x10$^{-3}$')
plt.subplots_adjust(wspace=0.025, hspace=0.005)
plt.savefig('../figures/s11_ridge_fingerprints.png', bbox_inches='tight')
plt.savefig('../figures/s11_ridge_fingerprints.pdf', bbox_inches='tight')
plt.show()




