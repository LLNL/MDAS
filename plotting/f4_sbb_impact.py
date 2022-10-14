#!/bin/env python
# -*- coding: utf-8 -*-

"""
Stephen Po-Chedley 27 August 2022

Create figure 4: effect of smoothed biomass burning

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
from fx import make_map, get_lat_lon_weight
from scipy import stats


# %% Parameters
fn_bbmaps = '../data/trendmaps/cesm2bb_ttt_trendmaps.pickle'
fn_ttt_all_trends = '../data/trends/cmip6_sat_era_ttt_trends.pickle'
fn_cesm2bb_trends = '../data/trends/cesm2bb_ttt_trends.pickle'
fn_obs_ttt_trends = '../data/trends/obs_ttt_trends.pickle'
fn_bcp_solution = '../data/predictions/overallPrediction_pls_tropical_1979-2014.pickle'
fn_color = '../data/metadata/colormap.pickle'
period = (1979, 2014)
domain = 'tropical'

# %% Load data
bbmaps = pickle.load(open(fn_bbmaps, 'rb'))
lat, lon, weight = get_lat_lon_weight()
alltrends = pickle.load(open(fn_ttt_all_trends, 'rb'))[domain]
cesm2bbtrends = pickle.load(open(fn_cesm2bb_trends, 'rb'))[domain]['CESM2BB']
obsTrends = pickle.load(open(fn_obs_ttt_trends, 'rb'))[domain]
bcp_solution = pickle.load(open(fn_bcp_solution, 'rb'))
color_map = pickle.load(open(fn_color, 'rb'))

# %% get mm averages
cesm2map = [bbmaps['CESM2'][key][period] for key in bbmaps['CESM2'].keys()]
cesm2map = np.mean(np.array(cesm2map), axis=0)
cesm2bbmap = [bbmaps['CESM2BB'][key][period] for key in bbmaps['CESM2BB'].keys()]
cesm2bbmap = np.mean(np.array(cesm2bbmap), axis=0)

# %% Set global font size
font = {'weight': 'normal',
        'size': 9}
plt.rc('font', **font)

# %% make figure
plt.figure(figsize=(8, 2.5), dpi=300)

# panel a
ax = plt.subplot(1, 2, 1, projection=ccrs.Robinson(central_longitude=180.))
bmap = cesm2bbmap - cesm2map
clevs = np.arange(-0.1, 0.01, 0.01)
im = make_map(ax, bmap, lat, lon, clevs, 'a. SBB TMT Trend Impact', cmap=plt.cm.Blues_r)
# colorbar
fig = plt.gcf()
cbar_ax = fig.add_axes([0.075, 0.1, 0.33, 0.045])
cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
fig.text(0.38, 0.16, '[K decade$^{-1}$]')

# %% panel b calculations
# get model distributions
cmip6trends = []
cmip6weights = []
cmip6enstrends = []
cesm2trends = []
c = 0
# loop over all models
for model in alltrends.keys():
    # loop over all members
    members = alltrends[model].keys()
    mtrends = []
    for member in members:
        t = alltrends[model][member]
        mtrends.append(t)
    # separate cesm2 (this distribution is plotted separately)
    # all others go in total distribution with ensemble weights
    if model == 'CESM2':
        cesm2trends = mtrends
    else:
        cmip6trends = cmip6trends + mtrends
        cmip6weights = cmip6weights + list(np.ones(len(mtrends))/len(mtrends))
        # keep track of ensemble mean values, too
        cmip6enstrends.append(np.mean(mtrends))
# get cesm2bb trends
cesm2bbtrends = [cesm2bbtrends[member][period][0] for member in cesm2bbtrends.keys()]

# do panel b plotting
plt.subplot(1, 2, 2)
# PDF of CMIP6 trends
kernel = stats.gaussian_kde(cmip6trends, weights=cmip6weights)
x_dist = np.arange(0, 0.6, 0.01)
y = kernel(x_dist)
plt.plot(x_dist, y, color='k', linewidth=1.5)
plt.plot(cmip6enstrends, np.zeros(len(cmip6enstrends)), '|', color='k')
plt.plot(np.mean(cmip6enstrends), 0, 'kx', markersize=10)
# cesm2 + bb
parts = plt.violinplot(cesm2trends, positions=[4], widths=[0.5], vert=False, showmeans=True)
cesm2color = color_map['CESM2']
for pc in parts['bodies']:
    pc.set_facecolor(cesm2color)
    pc.set_edgecolor(cesm2color)
    pc.set_alpha(1)
parts['cmaxes'].set_alpha(0)
parts['cmins'].set_alpha(0)
parts['cbars'].set_alpha(0)
parts['cmeans'].set_color(cesm2color)
parts['cmeans'].set_alpha(1)
c2m = np.mean(cesm2trends)
plt.plot([c2m, c2m], [4, 6.5], '--', color=cesm2color) # [3, 4]
# cesm2 bb distribution
parts = plt.violinplot(cesm2bbtrends, positions=[2], widths=[0.5], vert=False, showmeans=True)
for pc in parts['bodies']:
    pc.set_facecolor(cesm2color)
    pc.set_edgecolor(cesm2color)
    pc.set_alpha(1)
parts['cmaxes'].set_alpha(0)
parts['cmins'].set_alpha(0)
parts['cbars'].set_alpha(0)
parts['cmeans'].set_alpha(0)
c2bm = np.mean(cesm2bbtrends)
plt.plot([c2bm, c2bm], [2, 6.5], '--', color=cesm2color, alpha=1) # [2.42, 3]
# show cesm2bb - cesm2
plt.arrow(c2m, 6.5, c2bm-c2m, 0, head_width=0.25, head_length=0.01, length_includes_head=True, color='k', zorder=200)
# plot observational ranges
olist = [obsTrends[ds][period] for ds in obsTrends.keys()]
plt.axvspan(np.min(olist), np.max(olist), ymin=0.1, ymax=0.7, color='purple', alpha=0.25, linewidth=0)
# adjusted obs range
median_adjust = bcp_solution['unforced']['prediction']
olist_adjust = np.array(olist) - median_adjust
plt.axvspan(np.min(olist_adjust), np.max(olist_adjust), ymin=0.1, ymax=0.75, color='red', alpha=0.25, linewidth=0)
# show internal variability arrow
plt.arrow(np.mean(olist), 6.1, -median_adjust, 0, head_width=0.25, head_length=0.01, length_includes_head=True, color='k')
# ticks
ax = plt.gca()
xticks = np.arange(0, 0.61, 0.1)
yticks = np.arange(0, 8, 1)
yts = yticks[1] - yticks[0]
xts = xticks[1] - xticks[0]
yticksm = (np.arange(yticks[0] + yts/2, yticks[-1], yts))
xticksm = (np.arange(yticks[0] + yts/2, yticks[-1], xts))
ax.set_xticks(xticks)
ax.set_xticks(xticksm, minor=True)
ax.set_yticks(yticks)
ax.set_yticks(yticksm, minor=True)
# limits
plt.xlim([0, 0.6])
plt.ylim([-1, 8])
# spines
ax.tick_params(axis='both', which='major')
ax.tick_params(axis='both', which='minor')
ax.spines.left.set_position(('data', np.min(xticks) - 0.02))
ax.spines.left.set_bounds((np.min(yticks), np.max(yticks)))
ax.spines.right.set_color('none')
ax.spines.bottom.set_position(('data', -0.5))
ax.spines.bottom.set_bounds((np.min(xticks), np.max(xticks)))
ax.spines.top.set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
# labels
plt.xlabel('Tropical TMT Trend [K decade$^{-1}$]')
plt.text(0., 8., 'b. Influence of forcing and variability', fontsize=10)
plt.text(0.08, 0.4, 'Observed trend\n  range', rotation=90, color='purple')
plt.text(0.4, 1.2, 'CMIP6\n     distribution', rotation=0, color='k')
plt.text(0.25, 0.9, 'CESM2-SBB', color=cesm2color, alpha=1)
plt.text(0.4, 4.35, 'CESM2', color=cesm2color, alpha=1)
plt.text(0.3, 6.7, 'SBB sensitivity')
plt.text(0.085, 5.8, 'Internal variability\neffect')
# save
plt.tight_layout()
plt.savefig('../figures/f4_sbb_impact.pdf', bbox_inches='tight')
plt.savefig('../figures/f4_sbb_impact.png', bbox_inches='tight')
plt.show()

print('BB Impact: ', np.mean(cesm2bbtrends) - np.mean(cesm2trends))