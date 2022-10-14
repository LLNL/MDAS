#!/bin/env python
# -*- coding: utf-8 -*-

"""
Stephen Po-Chedley 21 September 2022

Create Table S1: models used in study

Copyright 2022, Stephen Po-Chedley, All rights reserved.

@author: pochedls
"""

# %% imports
import pickle
import numpy as np

# %% Parameters
domain = 'tropical'
period = (1979, 2014)

# %% specify / construct filenames
speriod = str(period[0]) + '-' + str(period[1])
fn_ttt_trends = '../data/trends/cmip6_ttt_trends.pickle'
fn_ecs = '../data/metadata/ecs.pickle'

# %% load data
trends = pickle.load(open(fn_ttt_trends, 'rb'))[domain]
ecs_data = pickle.load(open(fn_ecs, 'rb'))

# %% generate table
models = list(trends.keys())
models.sort()

table = []
table.append(['Model', 'n', 'ECS [K]'])
for model in models:
    n = len(trends[model].keys())
    ecs = ecs_data[model]
    ecs = '{0:.1f}'.format(np.round(ecs, 1))
    row = [model, n, ecs]
    table.append(row)

np.savetxt('../figures/table1.csv', table, delimiter=",", fmt='%s')
