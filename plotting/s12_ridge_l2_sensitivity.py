#!/bin/env python
# -*- coding: utf-8 -*-

"""
Stephen Po-Chedley 27 August 2022

Create figure S10: Ridge alpha sensitivity

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
periods = (1979, 2014)
alphavalues = [1, 128, 1000, 17500, 100000]
ptypes = ['unforced', 'forced']
fn_ridge_tuning = '../data/tuning/ridge.pickle'

# %% load data
ridge_model = pickle.load(open(fn_ridge_tuning, 'rb'))
beta = ridge_model['BETA']
lat, lon, weight = get_lat_lon_weight()
weight = np.squeeze(weight)

# %% Set global font size
font = {'weight': 'normal',
        'size': 9}
plt.rc('font', **font)

# %% make plot
# define maps to plot
plt.figure(figsize=(8, 10), dpi=300)
clevs = np.arange(-2.5, 2.51, 0.25)
# loop over ptypes and alpha values
for i, ptype in enumerate(ptypes):
    for j, alpha in enumerate(alphavalues):
        refmap = [beta[alpha][model][i] for model in beta[alpha].keys()]
        refmap = np.mean(np.array(refmap), axis=0)
        rvec = []
        for model in beta[alpha].keys():
            mmap = beta[alpha][model][i]
            r = weighted_corrcoef(np.reshape(refmap, -1), np.reshape(mmap, -1), weight)
            rvec.append(r)
        print('Minimum r-value', ptype, alpha, np.min(rvec))
        ax = plt.subplot(len(alphavalues), len(ptypes), j*2+i+1, projection=ccrs.Robinson(central_longitude=180.))
        im = make_map(ax, refmap*1000, lat, lon, clevs, '')
        if i == 0:
            ax.text(-0.07, 0.55, '$\\alpha$ = ' + str(alpha), va='bottom', ha='center', rotation='vertical', rotation_mode='anchor', transform=ax.transAxes)
        if j == 0:
            plt.title(ptype.capitalize())
# colorbar
fig = plt.gcf()
cbar_ax = fig.add_axes([0.12, 0.05, 0.8, 0.02])
cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
cbar.set_ticks(clevs[::4])
fig.text(0.1, 0.08, 'x10$^{-3}$')
plt.subplots_adjust(wspace=-0.1, hspace=0.08)
plt.savefig('../figures/s12_ridge_alpha_sensitivity.png', bbox_inches='tight')
plt.savefig('../figures/s12_ridge_alpha_sensitivity.pdf', bbox_inches='tight')
plt.show()

