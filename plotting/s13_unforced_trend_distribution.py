#!/bin/env python
# -*- coding: utf-8 -*-

"""
Stephen Po-Chedley 28 August 2022

Create figure S11: Plot unforced trend distributions

Copyright 2022, Stephen Po-Chedley, All rights reserved.

@author: pochedls
"""

# %% imports
import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# %% parameters
period = (1979, 2014)
domain = 'tropical'
mlmethod = 'pls'

# %% specify files
speriod = str(period[0]) + '-' + str(period[1])
fnhist = '../data/trends/cmip6_ttt_trends.pickle'
fnpic = '../data/trends/cmip6_ttt_picontrol_trends.pickle'
fn_bcp_solution = '../data/predictions/overallPrediction_' + mlmethod + '_' + domain + '_' + speriod + '.pickle'

# %% load data
trendsHistorical = pickle.load(open(fnhist, 'rb'))[domain]
trendsControl = pickle.load(open(fnpic, 'rb'))[domain]
bcp_solution = pickle.load(open(fn_bcp_solution, 'rb'))

# predicted unforced trends
median_adjust = bcp_solution['unforced']['prediction']
upper = bcp_solution['unforced']['upper']
lower = bcp_solution['unforced']['lower']

# %% Set global font size
font = {'weight': 'normal',
        'size': 9}
plt.rc('font', **font)

# %% create plot
histall = []
histweights = []
picall = []
piweights = []
models = list(trendsHistorical.keys())
plt.figure(figsize=(7, 10), dpi=300)
# plot observational ranges
# adjusted obs range
plt.axvspan(lower, upper, color='blue', alpha=0.25, linewidth=0)
plt.axvline(median_adjust, color='blue', linestyle='--')
plt.axvline(0, linestyle=':', color='k')

# loop over and plot models
for i, model in enumerate(models):
    # get historical data
    members = list(trendsHistorical[model].keys())
    periods = trendsHistorical[model][members[0]].keys()
    mhtrends = []
    for member in members:
        etrends = [trendsHistorical[model][member][period][2] for period in periods]
        mhtrends = mhtrends + etrends
    histall = histall + mhtrends
    histweights = histweights + list(np.ones(len(mhtrends)) / len(mhtrends))
    plt.plot(mhtrends, np.ones(len(mhtrends))*-i*3-0.5, '|', color='r')
    plt.text(0.18, -i*3-0.2-0.75, model)
    # number of members exceeding predicted unforced variability
    inds = np.where(mhtrends <= median_adjust)[0]
    ntotal = len(mhtrends)
    f = "{:.1f}".format(np.round(len(inds)/ntotal*100, 1)) + '% / ' + str(ntotal)
    plt.text(0.37, -i*3-0.42, f, color='r')
    # get picontrol data
    if model not in trendsControl:
        continue
    member = list(trendsControl[model].keys())[0]
    periods = trendsControl[model][member].keys()
    mptrends = [trendsControl[model][member][period][2] for period in periods]
    picall = picall + mptrends
    piweights = piweights + list(np.ones(len(mptrends)) / len(mptrends))
    plt.plot(mptrends, np.ones(len(mptrends))*-i*3-1.25, '|', color='gray')
    # number of members exceeding predicted unforced variability
    inds = np.where(mptrends <= median_adjust)[0]
    ntotal = len(mptrends)
    # check if any mean trends > 0.005 K/decade
    if ((np.abs(np.mean(mptrends)) > 0.005) | ((np.abs(np.mean(mhtrends)) > 0.005))):
        print(np.mean(mptrends), np.mean(mhtrends))
    f = "{:.1f}".format(np.round(len(inds)/ntotal*100, 1)) + '% / ' + str(ntotal)
    plt.text(0.37, -i*3-1.25, f, color='gray')

# PDF of all trends
# picontrol
kernel = stats.gaussian_kde(picall, weights=piweights)
x_dist = np.arange(-0.2, 0.2, 0.001)
y = kernel(x_dist)
plt.plot(x_dist, y, 'gray')

# historical
kernel = stats.gaussian_kde(histall, weights=histweights)
x_dist = np.arange(-0.2, 0.2, 0.001)
y = kernel(x_dist)
plt.plot(x_dist, y, 'red')

# Specify tick label size
ax = plt.gca()
xticks = np.arange(-0.2, 0.21, 0.05)
yticks = np.arange(0, 13, 3)
ax.tick_params(axis='both', which='major')
ax.set_xticks(xticks)
ax.set_yticks(yticks)

ax.spines.left.set_position(('data', np.min(xticks)-0.02))
ax.spines.right.set_color('none')
ax.spines.bottom.set_position(('data', np.min(xticks)-42))
ax.spines.top.set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.spines.left.set_bounds((0, 12))
ax.spines.bottom.set_bounds((np.min(xticks), np.max(xticks)))

plt.ylabel('Probability density     ', loc='top')
plt.xlabel('Unforced Tropical TMT Trend [K decade$^{-1}$]', loc='left')

plt.ylim(-41.5, 12)
plt.xlim(-0.2, 0.5)
plt.savefig('../figures/s13_variability_extremeness.pdf', bbox_inches='tight')
plt.savefig('../figures/s13_variability_extremeness.png', bbox_inches='tight')
plt.show()

# %% overall ratios
inds = np.where(picall < median_adjust)[0]
ntotal = len(picall)
f = "{:.1f}".format(np.round(len(inds)/ntotal*100, 1)) + '% / ' + str(ntotal)
print('piControl fraction: ' + f)

inds = np.where(histall < median_adjust)[0]
ntotal = len(histall)
f = "{:.1f}".format(np.round(len(inds)/ntotal*100, 1)) + '% / ' + str(ntotal)
print('historical fraction: ' + f)


