#!/bin/env python
# -*- coding: utf-8 -*-

"""
Stephen Po-Chedley 24 August 2022

Figure S2: Do disentanglement for all methods.

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
period = (1979, 2014)
obsColors = {'HadCRUT4-UAH': 'b', 'ERA5': 'r', 'GISTEMP': 'c', 'BEST': 'purple', 'HadCRUT5': 'sienna'}

# %% specify / construct filenames
speriod = str(period[0]) + '-' + str(period[1])
fn_ttt_trends = '../data/trends/cmip6_ttt_trends.pickle'
fn_ecs = '../data/metadata/ecs.pickle'
fn_color = '../data/metadata/colormap.pickle'
fn_obs_ttt_trends = '../data/trends/obs_ttt_trends.pickle'
fn_marker = '../data/metadata/markermap.pickle'

# %% load data
trends = pickle.load(open(fn_ttt_trends, 'rb'))[domain]
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

# %% overall plot options
methodLabels = ['Ridge Regression', 'PLS Regression', 'Neural Network']
plot_options = {'legend': {'on': False,
                           'x': 0.19,
                           'y': 0.10,
                           'vs': 0.02,
                           'hs': 0.02,
                           'th': 0.01},
                'limits': {'x': [-0.2, 0.2],
                           'y': [-0.3, 0.2]},
                'smarker': True,
                'xlabel': 'Predicted Trend [K decade$^{-1}$]',
                'ylabel': 'Actual Trend [K decade$^{-1}$]',
                'title': '',
                'xticks': np.arange(-0.2, 0.21, 0.1),
                'yticks': np.arange(-0.2, 0.21, 0.1),
                'spine_offset': [-0.02, -0.02],
                'obs_y_location': [-0.2, 0.022],
                'obslabels': False,
                'pdfheight': 0.05,
                'rlabel': [0, -0.125],
                'vlabel': [-0.17, 0.1],
                'xobs_shade': True}

# %% make figure
plt.figure(figsize=(8, 9), dpi=300)
for cn, mlmethod in enumerate(['ridge', 'pls', 'nn']):
    for rn, ptype in enumerate(['forced', 'unforced', 'total']):
        plt.subplot(4, 3, cn+rn*3+1)
        if ptype == 'forced':
            plot_options['smarker'] = False
            plot_options['limits'] = {'x': [0.1, 0.5], 'y': [0.1, 0.5]}
            plot_options['ylabel'] = 'Forced Tropical TMT\nTrend [K decade$^{-1}$]'
            tm = np.arange(0.1, 0.51, 0.1)
            plot_options['xticks'] = tm
            plot_options['yticks'] = tm
            plot_options['obs_y_location'] = [0.15, 0.009]
            plot_options['rlabel'] = [0.35, 0.15]
            plot_options['vlabel'] = [0.125, 0.45]
            plot_options['rline'] = True
            plot_options['obstrends'] = None
        elif ptype == 'unforced':
            plot_options['smarker'] = True
            plot_options['limits'] = {'x': [-0.2, 0.2], 'y': [-0.2, 0.2]}
            plot_options['ylabel'] = 'Unforced Tropical TMT\nTrend [K decade$^{-1}$]'
            tm = np.arange(-0.2, 0.21, 0.1)
            plot_options['xticks'] = tm
            plot_options['yticks'] = tm
            plot_options['obs_y_location'] = [-0.15, 0.009]
            plot_options['rlabel'] = [0, -0.125]
            plot_options['vlabel'] = [-0.17, 0.14]
            plot_options['rline'] = False
            plot_options['obstrends'] = None
        else:
            plot_options['smarker'] = True
            plot_options['limits'] = {'x': [0., 0.6], 'y': [0., 0.6]}
            plot_options['ylabel'] = 'Total Tropical TMT\nTrend [K decade$^{-1}$]'
            tm = np.arange(0., 0.61, 0.1)
            plot_options['xticks'] = tm
            plot_options['yticks'] = tm
            plot_options['obs_y_location'] = [0.07, 0.01]
            plot_options['rlabel'] = [0.3, 0.05]
            plot_options['vlabel'] = [0.05, 0.5]
            plot_options['rline'] = False
            plot_options['obstrends'] = obsTrends
        # load data based on prediction type
        fn_ttt_predictions = '../data/predictions/trendPredict_' + mlmethod + '_' + domain + '_' + speriod + '.pickle'
        fn_ttt_obs_predictions = '../data/predictions/obsPredict_' + mlmethod + '_' + domain + '_' + speriod + '.pickle'
        trendPredict = pickle.load(open(fn_ttt_predictions, 'rb'))
        obsPredict = pickle.load(open(fn_ttt_obs_predictions, 'rb'))
        make_scatter_plot(trends, trendPredict, obsPredict, marker_map, color_map, obsColors, period, ptype, models, plot_options)
        if rn == 0:
            plt.title(methodLabels[cn], loc='left')
# %%
plt.subplot2grid((4, 3), (3, 0), colspan=3)
xalign = [0, 0.4, 0.85]
nhalf = int(len(models)/2)
for i, model in enumerate(models):
    n = len(trendPredict[model].keys())
    c = color_map[model]
    m = marker_map[model]
    if i < nhalf:
        x = xalign[0]
        y = -i
    else:
        x = xalign[1]
        y = -i + nhalf
    plt.plot(x, y, marker=m, color=c, markersize=5)
    s = model + ' (' + str(n) + ')'
    plt.text(x+0.02, y-0.25, s, color=c)
plt.text(0, 1, 'Models', color='k', fontsize=11)
for i, dset in enumerate(['BEST', 'ERA5', 'GISTEMP', 'HadCRUT4-UAH', 'HadCRUT5']):
    y = -i
    c = obsColors[dset]
    plt.plot([xalign[2]-0.02, xalign[2]+0.02], [y, y], color=c)
    plt.text(xalign[2]+0.04, y-0.25, dset, color=c)
plt.text(xalign[2], 1, 'Observations', color='k', fontsize=11)
ax = plt.gca()
ax.axis('off')
plt.xlim(-0.1, 1.5)
plt.ylim(-nhalf-2, 2)
plt.tight_layout()
plt.savefig('../figures/s2_disentanglement_allmethods.pdf', bbox_inches='tight')
plt.savefig('../figures/s2_disentanglement_allmethods.png', bbox_inches='tight')
plt.show()
