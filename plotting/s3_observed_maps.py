#!/bin/env python
# -*- coding: utf-8 -*-

"""
Stephen Po-Chedley 27 August 2022

Create figure S3: observed warming

Copyright 2022, Stephen Po-Chedley, All rights reserved.

@author: pochedls
"""

# %% imports
import pickle
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy
import numpy as np
import sys
sys.path.append('..')
from fx import get_lat_lon_weight, weighted_corrcoef, get_obs_predictor_data, make_map


# %% Parameters
periods = [(1979, 2014), (1979, 2021)]
obsdatasets = ['BEST', 'ERA5', 'GISTEMP', 'HadCRUT4-UAH', 'HadCRUT5']
fn_obs = '../data/trendmaps/obs_tas_trendmaps.pickle'

# %% load data
obsTrends = pickle.load(open(fn_obs, 'rb'))
lat, lon, weight = get_lat_lon_weight()

# %% Set global font size
font = {'weight': 'normal',
        'size': 9}
plt.rc('font', **font)

# %% make plot
# define maps to plot
plt.figure(figsize=(8, 10), dpi=300)
clevs = np.arange(-0.6, 0.61, 0.05)
for i, period in enumerate(periods):
    for j, ds in enumerate(obsdatasets):
        dsmean = []
        members = list(obsTrends[ds].keys())
        if period not in obsTrends[ds][members[0]].keys():
            continue
        for member in members:
            dsmean.append(obsTrends[ds][member][period])
        bmap = np.mean(dsmean, axis=0)
        ax = plt.subplot(len(obsdatasets), len(periods), j*2+i+1, projection=ccrs.Robinson(central_longitude=180.))
        # kludge to gray out missing data
        # ax.fill_between([-10e200, 10e200], [-10e200, 10e200], color='gray')
        ax.add_feature(cartopy.feature.OCEAN, facecolor='gray')
        ax.add_feature(cartopy.feature.LAND, facecolor='gray')
        im = make_map(ax, bmap, lat, lon, clevs, '')
        if i == 0:
            ax.text(-0.07, 0.55, ds, va='bottom', ha='center', rotation='vertical', rotation_mode='anchor', transform=ax.transAxes)
        if j == 0:
            ptitle = str(period[0]) + '-' + str(period[1])
            plt.title(ptitle)
# colorbar
fig = plt.gcf()
cbar_ax = fig.add_axes([0.12, 0.05, 0.8, 0.02])
cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
cbar.set_ticks(clevs[::4])
fig.text(0.085, 0.08, '[K decade$^{-1}$]')
plt.subplots_adjust(wspace=-0.1, hspace=0.08)
plt.savefig('../figures/s3_observed_warming.png', bbox_inches='tight')
plt.savefig('../figures/s3_observed_warming.pdf', bbox_inches='tight')
plt.show()




