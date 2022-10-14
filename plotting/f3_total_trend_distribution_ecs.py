#!/bin/env python
# -*- coding: utf-8 -*-

"""
Stephen Po-Chedley 25 August 2022

Create figure 3: total trend prediction, histogram + adjustment,
                 ecs + adjustment

Copyright 2022, Stephen Po-Chedley, All rights reserved.

@author: pochedls
"""

# %% imports
import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import sys
sys.path.append('..')
from fx import make_scatter_plot, orthoregress


# %% Parameters
domain = 'tropical'
mlmethod = 'pls'
n_members = 10
period = (1979, 2014)
obsColors = {'BEST': 'purple', 'ERA5': 'r', 'GISTEMP': 'c', 'HadCRUT4-UAH': 'b', 'HadCRUT5': 'sienna'}

# %% specify / construct filenames
speriod = str(period[0]) + '-' + str(period[1])
fn_ttt_trends = '../data/trends/cmip6_ttt_trends.pickle'
fn_trend_maps = '../data/trendmaps/cmip6_tas_trendmaps.pickle'
fn_obs_trend_maps = '../data/trendmaps/obs_tas_trendmaps.pickle'
fn_ttt_all_trends = '../data/trends/cmip6_sat_era_ttt_trends.pickle'
fn_bcp_solution = '../data/predictions/overallPrediction_pls_tropical_1979-2014.pickle'
fn_ttt_predictions = '../data/predictions/trendPredict_' + mlmethod + '_' + domain + '_' + speriod + '.pickle'
fn_ttt_obs_predictions = '../data/predictions/obsPredict_' + mlmethod + '_' + domain + '_' + speriod + '.pickle'
fn_obs_ttt_trends = '../data/trends/obs_ttt_trends.pickle'
fn_ecs = '../data/metadata/ecs.pickle'
fn_color = '../data/metadata/colormap.pickle'
fn_marker = '../data/metadata/markermap.pickle'

# %% load data
trends = pickle.load(open(fn_ttt_trends, 'rb'))[domain]
trend_maps = pickle.load(open(fn_trend_maps, 'rb'))
obs_trend_maps = pickle.load(open(fn_obs_trend_maps, 'rb'))
alltrends = pickle.load(open(fn_ttt_all_trends, 'rb'))[domain]
trendPredict = pickle.load(open(fn_ttt_predictions, 'rb'))
bcp_solution = pickle.load(open(fn_bcp_solution, 'rb'))
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

plt.figure(figsize=(8, 4), dpi=300)
plt.subplot(1, 3, 1)
plot_options = {'legend': {'on': False,
                           'x': 0.19,
                           'y': 0.10,
                           'vs': 0.02,
                           'hs': 0.02,
                           'th': 0.01},
                'limits': {'x': [0.0, 0.6],
                           'y': [-0.15, 0.6]},
                'smarker': True,
                'rline': True,
                'xlabel': 'Sum (Forced + Unforced)\nPredicted Trend [K decade$^{-1}$]',
                'ylabel': 'Actual Trend [K decade$^{-1}$]',
                'title': 'a. Total Tropical TMT Trend',
                'xticks': np.arange(0.0, 0.61, 0.1),
                'yticks': np.arange(0.0, 0.61, 0.1),
                'spine_offset': [-0.02, -0.02],
                'obs_y_location': [-0.05, 0.015],
                'obs_xl_location': 0.15,
                'obslabels': False,
                'pdfheight': 0.05,
                'rlabel': [0.075, 0.5],
                'vlabel': [0.05, 0.35],
                'obstrends': obsTrends,
                'xobs_shade': True}

ptype = 'total'
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
plt.text(0.515, 0.55, '1 : 1', rotation=48)

# %% Panel B
plt.subplot(1, 3, 2)

histogram_values = []
colors = []
alltrendvalues = []
for model in models:
    mtrends = []
    members = list(alltrends[model].keys())
    for member in members:
        mtrends.append(alltrends[model][member])
    alltrendvalues = alltrendvalues + mtrends
    histogram_values.append(mtrends)
    colors.append(color_map[model])
othertrends = []
ecs_vector = []
gray_model_count = 0
for model in alltrends.keys():
    if model not in models:
        gray_model_count += 1
        members = list(alltrends[model].keys())
        for member in members:
            othertrends.append(alltrends[model][member])
alltrendvalues = alltrendvalues + othertrends
print(str(gray_model_count) + ' other models in gray')
histogram_values.append(othertrends)
colors.append('gray')
x = np.arange(0.0124, 0.6, 0.025)
plt.hist(histogram_values, bins=x, stacked=True, color=colors, orientation='horizontal')
# obs range
olist = [obsTrends[ds][period] for ds in obsTrends.keys()]
plt.axhspan(np.min(olist), np.max(olist), color='purple', alpha=0.25, linewidth=0)
# adjusted obs range
median_adjust = bcp_solution['unforced']['prediction']
upper_adjust = bcp_solution['unforced']['upper']
lower_adjust = bcp_solution['unforced']['lower']
olist_adjust = np.array(olist) - median_adjust
plt.axhspan(np.min(olist_adjust), np.max(olist_adjust), color='red', alpha=0.25, linewidth=0)
plt.axhline(np.min(olist)-upper_adjust, linestyle=':', color='r')
plt.axhline(np.max(olist)-lower_adjust, linestyle=':', color='r')
# adjustment arrow
plt.arrow(68, np.min(olist), 0, -median_adjust, head_width=2, head_length=0.01, length_includes_head=True, color='k', zorder=200)
plt.text(62, np.min(olist)+0.01, 'Shift', rotation=90)

# ticks
ax = plt.gca()
xticks = np.arange(0, 71, 10)
yticks = np.arange(.0, 0.61, 0.1)
yts = yticks[1] - yticks[0]
xts = xticks[1] - xticks[0]
yticksm = (np.arange(yticks[0] + yts/2, yticks[-1], yts))
xticksm = (np.arange(yticks[0] + yts/2, yticks[-1], xts))
ax.set_xticks(xticks)
ax.set_xticks(xticksm, minor=True)
ax.set_yticks(yticks)
ax.set_yticks(yticksm, minor=True)
# limits
plt.xlim([0, 70])
plt.ylim([-0.15, 0.6])
# spines
ax.tick_params(axis='both', which='major')
ax.tick_params(axis='both', which='minor')
sox = plot_options['spine_offset'][0]
soy = plot_options['spine_offset'][1]
ax.spines.left.set_position(('data', np.min(xticks) - 0.02))
ax.spines.left.set_bounds((np.min(yticks), np.max(yticks)))
ax.spines.right.set_color('none')
ax.spines.bottom.set_position(('data', plot_options['limits']['y'][0] - 0.02))
ax.spines.bottom.set_bounds((np.min(xticks), np.max(xticks)))
ax.spines.top.set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
# labels
plt.xlabel('Number of Simulations')
plt.title('b. CMIP6 Trend Distribution', fontsize=10)
ax.add_patch(Rectangle((7, -0.055), 7, 0.035, alpha=0.25, color='purple', edgecolor=None, linewidth=0))
ax.add_patch(Rectangle((7, -0.105), 7, 0.035, alpha=0.25, color='r', edgecolor=None, linewidth=0))
plt.text(15, -0.05, 'Observed Trend Range', color='purple', fontsize=8)
plt.text(15, -0.1, 'Internal variability removed', color='red', fontsize=8)

# %% ecs
plt.subplot(1, 3, 3)
x = []
y = []
cvalues = []
mvalues = []
orange = []
arange = []
for model in alltrends.keys():
    members = alltrends[model].keys()
    if model not in ecs_data.keys():
        continue
    mt = []
    for member in members:
        mt.append(alltrends[model][member])
    y1 = np.mean(mt)
    x1 = ecs_data[model]
    x.append(x1)
    y.append(y1)
    # get model ecs values in range of obs
    if ((y1 >= np.min(olist)) & (y1 <= np.max(olist))):
        orange.append(x1)
    # get model ecs values in range of obs
    if ((y1 >= np.min(olist_adjust)) & (y1 <= np.max(olist_adjust))):
        arange.append(x1)
    if model in marker_map.keys():
        c = color_map[model]
        m = marker_map[model]
    else:
        c = 'gray'
        m = 'o'
    plt.plot(x1, y1, marker=m, color=c, markersize=4)
# regression line
m, b, _, _, _ = orthoregress(x, y)
x1 = np.arange(np.min(x), np.max(x), 0.01)
plt.plot(x1, x1*m+b, 'k', linewidth=2)
r = np.corrcoef(x, y)[0, 1]
rs = '{:.2f}'.format(np.round(r, 2))
plt.text(2, 0.5, 'r = ' + rs)
# obs colors
plt.axhspan(np.min(olist), np.max(olist), color='purple', alpha=0.25, linewidth=0)
plt.axhspan(np.min(olist_adjust), np.max(olist_adjust), color='red', alpha=0.25, linewidth=0)
plt.axhline(np.min(olist)-upper_adjust, linestyle=':', color='r')
plt.axhline(np.max(olist)-lower_adjust, linestyle=':', color='r')
# plot model range
plt.plot([np.min(orange), np.max(orange)], [-0.105, -0.105], color='purple', linewidth=3)
plt.plot([np.min(arange), np.max(arange)], [-0.07, -0.07], color='red', linewidth=3)
# adjustment arrow
plt.arrow(6.4, np.min(olist), 0, -median_adjust, head_width=0.125, head_length=0.01, length_includes_head=True, color='k', zorder=200)
plt.text(6., np.min(olist)+0.01, 'Shift', rotation=90)
# ticks
ax = plt.gca()
xticks = np.arange(1.5, 6.6, 1)
yticks = np.arange(.0, 0.61, 0.1)
yts = yticks[1] - yticks[0]
xts = xticks[1] - xticks[0]
yticksm = (np.arange(yticks[0] + yts/2, yticks[-1], yts))
xticksm = (np.arange(yticks[0] + yts/2, yticks[-1], xts))
ax.set_xticks(xticks)
ax.set_xticks(xticksm, minor=True)
ax.set_yticks(yticks)
ax.set_yticks(yticksm, minor=True)
# limits
plt.xlim([1.5, 6.5])
plt.ylim([-0.15, 0.6])
# spines
ax.tick_params(axis='both', which='major')
ax.tick_params(axis='both', which='minor')
ax.spines.left.set_position(('data', np.min(xticks) - 0.02))
ax.spines.left.set_bounds((np.min(yticks), np.max(yticks)))
ax.spines.right.set_color('none')
ax.spines.bottom.set_position(('data', plot_options['limits']['y'][0] - 0.02))
ax.spines.bottom.set_bounds((np.min(xticks), np.max(xticks)))
ax.spines.top.set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
# labels
plt.xlabel('ECS [K]')
plt.title('c. Tropical TMT Trend and ECS', fontsize=10)
plt.text(1.6, -0.05, 'ECS values consistent\n     with observed warming')

plt.tight_layout()
plt.savefig('../figures/f3_total_trend_distribution_ecs.pdf', bbox_inches='tight')
plt.savefig('../figures/f3_total_trend_distribution_ecs.png', bbox_inches='tight')
plt.show()

# %% fraction of simulations within range of observations
inds = np.where((alltrendvalues <= np.max(olist)) & (alltrendvalues >= np.min(olist)))[0]
print(str(len(inds)) + ' / ' + str(len(alltrendvalues)) + ' below MSU upper bound (' + str(len(inds) / len(alltrendvalues)) + ')')

inds = np.where((np.array(alltrendvalues) <= np.max(olist)-median_adjust))[0]
print(str(len(inds)) + ' / ' + str(len(alltrendvalues)) + ' below MSU upper bound after adjustment (' + str(len(inds) / len(alltrendvalues)) + ')')
