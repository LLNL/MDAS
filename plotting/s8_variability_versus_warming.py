#!/bin/env python
# -*- coding: utf-8 -*-

"""
Stephen Po-Chedley 30 August 2022

Maps showing ratio of internal variability to forced trend.

Copyright 2022, Stephen Po-Chedley, All rights reserved.

@author: pochedls
"""

# %% imports
import cartopy.crs as ccrs
import sys
sys.path.append('..')
from fx import get_lat_lon_weight, make_map
import pickle
import matplotlib.pyplot as plt
import numpy as np

# %% Parameters
period = (1979, 2014)
fntas = '../data/trendmaps/cmip6_tas_trendmaps.pickle'

# %% load data
trendMaps = pickle.load(open(fntas, 'rb'))
lat, lon, weight = get_lat_lon_weight()

# %% Set global font size
font = {'weight': 'normal',
        'size': 8}
plt.rc('font', **font)

# %% loop over model ensembles to compute warming and standard deviation
multimodel_warming = []
multimodel_std = []
for i, model in enumerate(trendMaps.keys()):
    modelmaps = []
    for member in trendMaps[model].keys():
        mmap = trendMaps[model][member][period]
        modelmaps.append(mmap)
    modelmaps = np.array(modelmaps)
    mmean = np.mean(modelmaps, axis=0)
    mstd = np.std(modelmaps, axis=0)
    multimodel_warming.append(mmean)
    multimodel_std.append(mstd)
multimodel_warming = np.array(multimodel_warming)
multimodel_std = np.array(multimodel_std)
multimodel_warming = np.mean(multimodel_warming, axis=0)
multimodel_std = np.mean(multimodel_std, axis=0)

# %%
cmap = plt.cm.Reds
cmapr = plt.cm.RdBu_r
plt.figure(figsize=(8, 6), dpi=300)
# panel a
ax = plt.subplot(2, 2, 1, projection=ccrs.Robinson(central_longitude=180.))
im = make_map(ax, multimodel_warming, lat, lon, np.arange(0, .91, 0.05), 'a. Simulated warming [K decade$^{-1}$]', cmap=cmap, extend='max')
cbar = plt.colorbar(location='bottom', ticks=[0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9])
# panel b
ax = plt.subplot(2, 2, 2, projection=ccrs.Robinson(central_longitude=180.))
im = make_map(ax, multimodel_std, lat, lon, np.arange(0, 0.36, 0.025), 'b. Standard deviation of warming [K decade$^{-1}$]', cmap=cmap, extend='max')
cbar = plt.colorbar(location='bottom', ticks=[0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35])
# panel c
ax = plt.subplot(2, 2, 3, projection=ccrs.Robinson(central_longitude=180.))
im = make_map(ax, np.log2(multimodel_warming / multimodel_std), lat, lon, np.arange(-3, 3.1, 1/3.), 'c. log$_2$ of the ratio of warming to variability', cmap=cmapr, extend='both')
cbar = plt.colorbar(location='bottom', ticks=[-3, -2, -1, 0, 1, 2, 3])
# panel d
ax = plt.subplot(2, 2, 4, projection=ccrs.Robinson(central_longitude=180.))
im = make_map(ax, np.log2(multimodel_std / multimodel_warming), lat, lon, np.arange(-3, 3.1, 1/3.), 'd. log$_2$ of the ratio of variability to warming', cmap=cmapr, extend='both')
cbar = plt.colorbar(location='bottom', ticks=[-3, -2, -1, 0, 1, 2, 3])

plt.savefig('../figures/s8_variability_versus_warming.pdf', bbox_inches='tight')
plt.savefig('../figures/s8_variability_versus_warming.png', bbox_inches='tight')
plt.show()


