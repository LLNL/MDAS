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
from fx import make_scatter_plot, get_observed_predictions, get_predicted_and_actual_trends, orthoregress


# %% Parameters
domain = 'tropical'
mlmethod = 'pls'
period = (1979, 2014)
obsColors = {'BEST': 'purple', 'ERA5': 'r', 'GISTEMP': 'c', 'HadCRUT4-UAH': 'b', 'HadCRUT5': 'sienna'}

# %% specify / construct filenames
speriod = str(period[0]) + '-' + str(period[1])
fn_ttt_trends = '../data/trends/cmip6_ttt_trends.pickle'
fn_ttt_predictions = '../data/predictions/trendPredict_' + mlmethod + '_' + domain + '_' + speriod + '.pickle'
fn_ttt_obs_predictions = '../data/predictions/obsPredict_' + mlmethod + '_' + domain + '_' + speriod + '.pickle'
fn_ecs = '../data/metadata/ecs.pickle'
fn_color = '../data/metadata/colormap.pickle'
fn_marker = '../data/metadata/markermap.pickle'

# %% load data
trends = pickle.load(open(fn_ttt_trends, 'rb'))[domain]
trendPredict = pickle.load(open(fn_ttt_predictions, 'rb'))
obsPredict = pickle.load(open(fn_ttt_obs_predictions, 'rb'))
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

plt.figure(figsize=(8, 4), dpi=300)
plt.subplot(1, 2, 1)
plot_options = {'legend': {'on': True,
                           'x': 0.19,
                           'y': 0.10,
                           'vs': 0.02,
                           'hs': 0.02,
                           'th': 0.01},
                'limits': {'x': [-0.2, 0.2],
                           'y': [-0.375, 0.2]},
                'smarker': True,
                'rline': True,
                'xlabel': 'Predicted Trend [K decade$^{-1}$]',
                'ylabel': 'Actual Trend [K decade$^{-1}$]',
                'title': 'a. Unforced Tropical TMT Trend',
                'xticks': np.arange(-0.2, 0.21, 0.1),
                'yticks': np.arange(-0.2, 0.21, 0.1),
                'spine_offset': [-0.02, -0.02],
                'obs_y_location': [-0.25, 0.022],
                'obs_xl_location': -0.1,
                'obslabels': True,
                'pdfheight': 0.075,
                'rlabel': [0, -0.125],
                'vlabel': [-0.17, 0.1],
                'obstrends': None,
                'xobs_shade': True}
ptype = 'unforced'
make_scatter_plot(trends, trendPredict, obsPredict, marker_map, color_map, obsColors, period, ptype, models, plot_options)
plt.text(0.14, 0.16, '1 : 1', rotation=44.5)

plt.subplot(1, 2, 2)
ptype = 'forced'
plot_options = {'legend': {'on': False,
                           'x': 0.19,
                           'y': 0.10,
                           'vs': 0.02,
                           'hs': 0.02,
                           'th': 0.01},
                'limits': {'x': [0.1, 0.5],
                           'y': [-0.075, 0.5]},
                'xlabel': 'Predicted Trend [K decade$^{-1}$]',
                'ylabel': 'Actual Trend [K decade$^{-1}$]',
                'smarker': False,
                'rline': True,
                'title': 'b. Forced Tropical TMT Trend',
                'xticks': np.arange(0.1, 0.51, 0.1),
                'yticks': np.arange(0.1, 0.51, 0.1),
                'spine_offset': [-0.02, -0.02],
                'obs_y_location': [0.05, 0.022],
                'obs_xl_location': 0.2,
                'obslabels': True,
                'pdfheight': 0.075,
                'rlabel': [0.3, 0.175],
                'vlabel': [0.125, 0.375],
                'obstrends': None,
                'xobs_shade': True}
make_scatter_plot(trends, trendPredict, obsPredict, marker_map, color_map, obsColors, period, ptype, models, plot_options)
plt.savefig('../figures/f1_disentanglement.pdf', bbox_inches='tight')
plt.savefig('../figures/f1_disentanglement.png', bbox_inches='tight')
plt.show()

# %% print raw observational predictions
for i, ds in enumerate(obsColors.keys()):
    yp = get_observed_predictions(obsPredict, period, [ds])
    if i == 0:
        ovalues = yp.copy()
    else:
        ovalues = np.concatenate((ovalues, yp), axis=0)
print('mean values: ', np.mean(ovalues, axis=0))
print('min values: ', np.min(ovalues, axis=0))
print('max values: ', np.max(ovalues, axis=0))

# %% get bias in unforced prediction plot
xall = []
yall = []
wall = []
for model in models:
    xm, ym = get_predicted_and_actual_trends(trends, trendPredict, model, period=period, ptype=['unforced'])
    xm = xm[:, 0]; ym = ym[:, 0]
    wm = np.ones(len(xm)) / len(xm)
    xall = xall + list(xm)
    yall = yall + list(ym)
    wall = wall + list(wm)
m, b, _, _, _ = orthoregress(xall, yall, weights=wall)
yp_min = np.min(xall)*m+b
yp_min_bias = np.min(xall) - yp_min
yp_max = np.max(xall)*m+b
yp_max_bias = np.max(xall) - yp_max
print('fitted bias at smallest / largest points: ' + str(yp_min_bias) + ' / ' + str(yp_max_bias))

# %% ckeck sensitivity to masking
print()
print('largest differences ERA masked versus unmasked')
dsdiffsall = []
for ds in ['BEST', 'GISTEMP', 'HadCRUT5']:
    print(ds)
    dsm = 'ERA5_masked_' + ds
    dsdiffs = []
    for model in obsPredict.keys():
        member = list(obsPredict[model][dsm].keys())[0]
        zm = obsPredict[model][dsm][member][period]
        zf = obsPredict[model]['ERA5'][member][period]
        d = zm - zf
        dsdiffs.append(d)
        dsdiffsall.append(d)
    dsdiffs = np.array(dsdiffs)
    for i, ptype in enumerate(['total', 'forced', 'unforced']):
        x = dsdiffs[:, i]
        inds = np.where(np.max(np.abs(x)) == np.abs(x))[0]
        print(ptype + ': ' + str(float(x[inds])))
dsdiffsall = np.array(dsdiffsall)
print('mean differences: ', np.mean(dsdiffsall[:, 1:3], axis=0))

