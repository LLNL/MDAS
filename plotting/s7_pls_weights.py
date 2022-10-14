#!/bin/env python
# -*- coding: utf-8 -*-

"""
Stephen Po-Chedley 27=8 August 2022

Create figure S6: PLS weights

Copyright 2022, Stephen Po-Chedley, All rights reserved.

@author: pochedls
"""

# %% imports
import pickle
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cartopy.crs as ccrs
import numpy as np
import sys
sys.path.append('..')
from fx import get_lat_lon_weight, weighted_corrcoef, make_map

# %% Parameters
fn_pls_model = '../data/mlmodels/plsModel_tropical_1979-2014.pickle'


# %% load data
pls_model = pickle.load(open(fn_pls_model, 'rb'))
lat, lon, weight = get_lat_lon_weight()
weight = np.squeeze(weight)

# %% Set global font size
font = {'weight': 'normal',
        'size': 9}
plt.rc('font', **font)

# %% first make sure maps all have the same x-y sign
models = list(pls_model['x_weights_'].keys())
n_components = pls_model['x_weights_'][models[0]].shape[0]
# loop over each component and model and align correlation
# with model 1
for i in range(n_components):
    map1 = pls_model['x_weights_'][models[0]][i]
    rvec = []
    for model in models:
        mmap = pls_model['x_weights_'][model][i]
        r = weighted_corrcoef(np.reshape(map1, -1), np.reshape(mmap, -1), weight)
        if r < 0:
            pls_model['x_weights_'][model][i] = -pls_model['x_weights_'][model][i]
            pls_model['y_weights_'][model][i] = -pls_model['y_weights_'][model][i]
            r = -r
        rvec.append(r)
    print('Minimum Correlation for Component ' + str(i+1) + ': ' + str(np.min(rvec)))

# %% make plot
# define maps to plot
fig = plt.figure(figsize=(8, 10), dpi=300)
gs = GridSpec(5, 2, width_ratios=[1, 1], height_ratios=[1, 1, 1, 0.5, 1])
clevs = np.arange(-0.02, 0.021, 0.002)
# loop over ptypes and alpha values
unforced_y_weights = []
forced_y_weights = []
for i, component in enumerate(range(n_components)):
    refmap = [pls_model['x_weights_'][model][i] for model in pls_model['x_weights_'].keys()]
    refmap = np.mean(np.array(refmap), axis=0)
    ufyw = [pls_model['y_weights_'][model][i, 0] for model in pls_model['y_weights_'].keys()]
    fyw = [pls_model['y_weights_'][model][i, 1] for model in pls_model['y_weights_'].keys()]
    unforced_y_weights.append(ufyw)
    forced_y_weights.append(fyw)
    ax = fig.add_subplot(gs[i], projection=ccrs.Robinson(central_longitude=180.))
    im = make_map(ax, refmap, lat, lon, clevs, 'Component ' + str(i+1))
# colorbar
fig = plt.gcf()
cbar_ax = fig.add_axes([0.135, 0.32, 0.74, 0.025])
cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
cbar.set_ticks(clevs[::4])
# bar plot
ax = fig.add_subplot(gs[4, :])
plt.plot([1, 2], [1, 2])

bw = 0.9 / len(models) / 2
for i in range(n_components):
    # forced values
    x = np.arange(0.5*bw, bw*(len(models)), bw) + i +bw*2
    y = forced_y_weights[i]
    plt.bar(x, y, width=bw, edgecolor='r', facecolor='w', linewidth=0.5)
    # unforced values
    x = np.arange(0.5*bw, bw*(len(models)), bw) + i +0.5 + bw
    y = unforced_y_weights[i]
    plt.bar(x, y, width=bw, edgecolor='b', facecolor='w', linewidth=0.5)

# ticks
ax = plt.gca()
xticks = np.arange(0.5, 6, 1)
plt.xticks(xticks, labels=np.arange(1, n_components+1))
yticks = np.arange(-0.02, 0.041, 0.02)
yts = yticks[1] - yticks[0]
xts = xticks[1] - xticks[0]
yticksm = (np.arange(yticks[0] + yts/2, yticks[-1], yts))
xticksm = (np.arange(yticks[0] + yts/2, yticks[-1], xts))
ax.set_xticks(xticks)
ax.set_xticks(xticksm, minor=True)
ax.set_yticks(yticks)
ax.set_yticks(yticksm, minor=True)
# limits
plt.xlim([0, 6.5])
plt.ylim([-0.03, 0.04])
# spines
ax.tick_params(axis='both', which='major')
ax.tick_params(axis='both', which='minor')
ax.spines.left.set_position(('data', - 0.02))
ax.spines.left.set_bounds((np.min(yticks), np.max(yticks)))
ax.spines.right.set_color('none')
ax.spines.bottom.set_position(('data', np.min(yticks)-0.01))
ax.spines.bottom.set_bounds((0, 6.2))
ax.spines.top.set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
plt.xlabel('Component')
plt.ylabel('Y-weight')
plt.text(0.2, 0.035, 'Forced', color='r')
plt.text(0.2, 0.0275, 'Unforced', color='blue')
plt.plot([0, 6.5], [0, 0], 'k', linewidth=0.5)
plt.savefig('../figures/s7_pls_components.png', bbox_inches='tight')
plt.savefig('../figures/s7_pls_components.pdf', bbox_inches='tight')
plt.show()

