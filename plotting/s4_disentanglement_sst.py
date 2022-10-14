#!/bin/env python
# -*- coding: utf-8 -*-

"""
Stephen Po-Chedley 24 August 2022

Create figure 1: forced and unforced predictions.

Copyright 2022, Stephen Po-Chedley, All rights reserved.

@author: pochedls
"""

# %% imports
import pickle
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('..')
from fx import make_scatter_plot


# %% Parameters
domain = 'tropical'
mlmethod = 'pls'
period = (1979, 2014)
obsColors = {'COBE2': 'r', 'ERSSTv5': 'c', 'HadISST1.1': 'b', 'HadSST4': 'k', 'PCMDI': 'sienna'}

# %% specify / construct filenames
speriod = str(period[0]) + '-' + str(period[1])
fn_ttt_trends = '../data/trends/cmip6_ttt_trends.pickle'
fn_ttt_predictions = '../data/predictions/trendPredict_' + mlmethod + '_' + domain + '_sst_' + speriod + '.pickle'
fn_ttt_obs_predictions = '../data/predictions/obsPredict_' + mlmethod + '_' + domain + '_sst_' + speriod + '.pickle'
fn_obs_ttt_trends = '../data/trends/obs_ttt_trends.pickle'
fn_ecs = '../data/metadata/ecs.pickle'
fn_color = '../data/metadata/colormap.pickle'
fn_marker = '../data/metadata/markermap.pickle'

# %% load data
trends = pickle.load(open(fn_ttt_trends, 'rb'))[domain]
trendPredict = pickle.load(open(fn_ttt_predictions, 'rb'))
obsPredict = pickle.load(open(fn_ttt_obs_predictions, 'rb'))
obsTrends = pickle.load(open(fn_obs_ttt_trends, 'rb'))[domain]
ecs_data = pickle.load(open(fn_ecs, 'rb'))
color_map = pickle.load(open(fn_color, 'rb'))
marker_map = pickle.load(open(fn_marker, 'rb'))
models = list(trends.keys())
# order models by ecs
models = list(np.array(models)[np.argsort([ecs_data[model] for model in models])])

# %% Set global font size
font = {'weight': 'normal',
        'size': 9}
plt.rc('font', **font)

# %% make figure

plt.figure(figsize=(4.75, 10), dpi=300)
plt.subplot(3, 1, 1)
plot_options = {'legend': {'on': True,
                           'x': 0.19,
                           'y': 0.10,
                           'vs': 0.025,
                           'hs': 0.02,
                           'th': 0.01},
                'limits': {'x': [-0.2, 0.2],
                           'y': [-0.3, 0.2]},
                'smarker': True,
                'rline': True,
                'xlabel': 'Predicted Trend [K decade$^{-1}$]',
                'ylabel': 'Actual Trend [K decade$^{-1}$]',
                'title': 'a. Unforced Tropical TMT Trend',
                'xticks': np.arange(-0.2, 0.21, 0.1),
                'yticks': np.arange(-0.2, 0.21, 0.1),
                'spine_offset': [-0.02, -0.02],
                'obs_y_location': [-0.175, 0.022],
                'obs_xl_location': -0.1,
                'obslabels': True,
                'pdfheight': 0.075,
                'rlabel': [-0.17, 0.06],
                'vlabel': [-0.17, 0.1],
                'obstrends': None,
                'xobs_shade': True}
ptype = 'unforced'
make_scatter_plot(trends, trendPredict, obsPredict, marker_map, color_map, obsColors, period, ptype, models, plot_options)
plt.text(0.14, 0.16, '1 : 1', rotation=44.5)

plt.subplot(3, 1, 2)
ptype = 'forced'
plot_options = {'legend': {'on': False,
                           'x': 0.19,
                           'y': 0.10,
                           'vs': 0.02,
                           'hs': 0.02,
                           'th': 0.01},
                'limits': {'x': [0.1, 0.5],
                           'y': [0.0, 0.5]},
                'xlabel': 'Predicted Trend [K decade$^{-1}$]',
                'ylabel': 'Actual Trend [K decade$^{-1}$]',
                'smarker': False,
                'rline': True,
                'title': 'b. Forced Tropical TMT Trend',
                'xticks': np.arange(0.1, 0.51, 0.1),
                'yticks': np.arange(0.1, 0.51, 0.1),
                'spine_offset': [-0.02, -0.02],
                'obs_y_location': [0.125, 0.022],
                'obs_xl_location': 0.2,
                'obslabels': False,
                'pdfheight': 0.075,
                'rlabel': [0.3, 0.175],
                'vlabel': [0.125, 0.375],
                'obstrends': None,
                'xobs_shade': True}
make_scatter_plot(trends, trendPredict, obsPredict, marker_map, color_map, obsColors, period, ptype, models, plot_options)

plt.subplot(3, 1, 3)
ptype = 'total'
plot_options = {'legend': {'on': False,
                           'x': 0.19,
                           'y': 0.10,
                           'vs': 0.02,
                           'hs': 0.02,
                           'th': 0.01},
                'limits': {'x': [0., 0.6],
                           'y': [-0.12, 0.6]},
                'xlabel': 'Predicted Trend [K decade$^{-1}$]',
                'ylabel': 'Actual Trend [K decade$^{-1}$]',
                'smarker': True,
                'rline': True,
                'title': 'c. Total Tropical TMT Trend',
                'xticks': np.arange(0.0, 0.61, 0.1),
                'yticks': np.arange(0., 0.61, 0.1),
                'spine_offset': [-0.02, -0.02],
                'obs_y_location': [0.025, 0.025],
                'obs_xl_location': 0.2,
                'obslabels': False,
                'pdfheight': 0.075,
                'rlabel': [0.35, 0.25],
                'vlabel': [0.05, 0.45],
                'obstrends': obsTrends,
                'xobs_shade': True}
make_scatter_plot(trends, trendPredict, obsPredict, marker_map, color_map, obsColors, period, ptype, models, plot_options)

# staggerer msu line labels
llength = [0.25, 0.3, 0.35, 0.4]
for i, dset in enumerate(['uah', 'uw', 'rss', 'noaa']):
    y = obsTrends[dset][period]
    x = llength[i]
    plt.plot([0, x], [y, y], color='purple', linewidth=0.75)
    if dset == 'rss':
        plt.text(x+0.02, y-0.015, 'UW / RSS', color='purple')
    elif dset == 'uw':
        continue
    else:
        plt.text(x+0.02, y-0.015, dset.upper(), color='purple')

plt.savefig('../figures/s4_disentanglement_sst.pdf', bbox_inches='tight')
plt.savefig('../figures/s4_disentanglement_sst.png', bbox_inches='tight')
plt.show()
