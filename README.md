# MSU Disentanglement Analysis Software (MDAS)

This software is used to disentangle the forced and unforced components of tropospheric temperature change over the satellite era (after 1979) using maps of surface temperature change as a predictor. In general, the software assembles training datasets (from pre-computed surface temperature trend maps and domain averaged tropospheric warming rates), trains statistical/machine learning (ML) algorithms, applies the trained statistical/ML model to climate model data and observations, and then saves the results. A leave-one-out approach is used in which the statistical/ML models are iteratively trained on (N-1) climate models and then applied to the remaining climate model (and observations). Each model includes a large ensemble of model simulations (i.e., >9 members). The software relies on scikit-learn ridge regression, PLS regression, and neural network algorithms (Pedregosa et al., 2011). 

### Accompanying Data and Manuscript

This software is accompanied by the input data, which is available via Zenodo (doi: [10.5281/zenodo.7199961](https://zenodo.org/record/7199961)). 

The research that this software underpins was documented in:

> Po-Chedley, S., J.T. Fasullo, N. Siler, Z.M. Labe, E.A. Barnes, C.J.W. Bonfils, B.D. Santer (2022): "Internal variability and forcing influence model-satellite differences in the rate of tropical tropospheric warming," Proceedings of the National Academy of Sciences, doi: 10.1073/pnas.2209431.

### Environment

This software uses Python 3 and the software environment can be created using Anaconda. The environment is also specified in environment.yml. It can be installed with:

```
conda env create -f environment.yml
conda activate xcdat
```

OR: 

```
conda create -n xcdat -c conda-forge xcdat scikit-learn scipy cftime matplotlib cartopy ipython
conda activate xcdat
```

### Organization

The software is organized as follows:
    
    environment.yml
        Conda environment used to produce code.

    tune_ml.py
        Python script to evaluate error associated with different ML parameters.
        Writes to data/tuning/.

    run_ml.py
        Python script to execute machine learning algorithms to produce predicted values for 
        forced-versus-unforced warming. Results are written to data/predictions/ and
        data/mlmodels/.

    fx.py
        Function library (see individual function documentation).

    fx_odr.py
        Function library for orthogonal distance regression.

    data/
        Directory containing data used by software.

    plotting/
        Directory contains scripts to produce individual manuscript figure files.

Note that scripts write to `data/mlmodels/`, `data/predictions/`, and `data/tuning/`. These directories should be created
before executing `run_ml.py` and `tune_ml.py`. Figures are saved to `figures/` in both pdf and png format.

After selecting ML parameters (informed by output from `tune_ml.py`) `run_ml.py` is used to train and execute ML predictions.
The `plotting/` directory then contains the files to produce individual manuscript figures.

### Data Description

Data that accompanies this software are derived from other publicly available datasets (see the accompanying publication or dataset on Zenodo for details). The data is principally composed of maps of temperature trends and values of area averaged temperature trends. We also include some data used in figures, including a dictionary of the markers and colors used for each climate model in figures and the effective climate sensitivity values derived from Zelinka et al. (2020). The data is organized into three main directories:

    trendmaps/
        cesm2bb_ttt_trendmaps.pickle
            Trend maps of TTT trends for a given model, ensemble member, and time period. 
            Structure: dataDictionary[model][member][period]
        cmip6_tas_spliced_trendmaps.pickle
            Trend maps of surface temperature (tas) trends for a given model, ensemble member,
            and time period for the historical experiment extended with scenario experiments (e.g.,
            SSP370 or SSP585). Structure: dataDictionary[model][member][period]
        cmip6_tas_trendmaps.pickle
            Trend maps of surface temperature (tas) trends for a given model, ensemble member, and
            time period for the historical experiment. Structure: dataDictionary[model][member][period]
        cmip6_tas_trendmaps.pickle
            Trend maps of surface skin temperature (ts) trends for a given model, ensemble member, and
            time period for the historical experiment. Structure: dataDictionary[model][member][period]
        obs_tas_trendmaps.pickle
            Observationally-derived trend maps of surface temperature trends for a given dataset,
            ensemble member (if applicable), and time period. 
            Structure: dataDictionary[dataset][member][period]
        obs_sst_trendmaps.pickle
            Observationally-derived trend maps of sea surface temperature trends for a given dataset,
            ensemble member (if applicable), and time period.
            Structure: dataDictionary[dataset][member][period]

    trends/
        cesm2bb_ttt_trends.pickle
            The TTT trends for the CESM2-SBB large ensemble.
            Structure: dataDictionary[domain][model][member][period]
        cmip6_ttt_picontrol_trends.pickle 
            The TTT trends for CMIP6 piControl simulations (the forced component of the trend is zero).
            Structure: dataDictionary[domain][model][member][period]
        cmip6_ttt_trends.pickle
            TTT trends for CMIP6 historical simulations.
            Structure: dataDictionary[domain][model][member][period]
        cmip6_sat_era_ttt_trends.pickle
            Total TTT trends for the time period 1979 through 2014 from CMIP6 models. 
            Structure: dataDictionary[domain][model][member]
        cmip6_ttt_spliced_trends.pickle
            TTT trends for CMIP6 historical simulations extended with scenario experiments (e.g., SSP370
            or SSP585). Structure: dataDictionary[domain][model][member][period]
        obs_ttt_trends.pickle
            Observationally-derived TTT trend values for a given dataset and time period. 
            Structure: dataDictionary[domain][dataset][period]

    metadata/
        colormap.pickle
            Color code for each model (in RGB format). Structure: dataDictionary[model]
        ecs.pickle
            Effective climate sensitivity value for each model. Structure: dataDictionary[model]
        markermap.pickle
            Matplotlib marker style for each model. Structure: dataDictionary[model]

Each file is a Python pickle file which contains dictionaries or nested dictionaries (structure given above). Pickle files can be opened using Python, typically with:
    
    import pickle
    data = pickle.load(open(filename.pickle, 'rb'))

The keys include the model/dataset name (string, e.g., 'CESM2'), the model ensemble member (string, e.g., 'r1i1p1f1'), the period (tuple, e.g., (1979, 2014)), and the domain (string, either 'tropical' or 'global'). The tropical domain spans 30S - 30N and the global domain is over 82.5S - 82.5N. 

The trend map data is on a 2.5 x 2.5 degree latitude x longitude grid (72 x 144 spanning -88.75 to 88.75 and 1.25 to 358.75, respectively). 

The trend values are typically in 3-element lists broken down into the total, forced, and unforced components of the trend. 

One exception is the file cmip6_sat_era_ttt_trends.pickle, which only contains the total trend because this file also includes models with insufficient ensemble members to compute the forced component of the trend. The same is true for obs_ttt_trends.pickle

Note that here we refer to corrected trends in the temperature of the mid-troposphere (TMT) as TTT.

### Acknowledgements

Research at Lawrence Livermore National Laboratory (LLNL) was performed under the auspices of US Department of Energy Contract DE-AC52-07NA27344. S.P. and C.J.W.B. were supported through the PCMDI Project, which is funded by the Regional and Global Model Analysis Program of the Office of Science at the US Department of Energy. 

### References:

F. Pedregosa, et al., Scikit-learn: Machine Learning in Python. J. Mach. Learn. Res. 12, 2825–2830 (2011).

Zelinka, M. D., T. A. Myers, D. T. McCoy, S. Po-Chedley, P. M. Caldwell, P. Ceppi, S. A. Klein, and K. E. Taylor, 2020: Causes of higher climate sensitivity in CMIP6 models, Geophys. Res. Lett., 47, doi:10.1029/2019GL085782.

### License

MDAS uses the MIT license.

See [LICENSE](https://github.com/LLNL/MDAS/blob/main/LICENSE)

SPDX-License-Identifier: MIT

LLNL-CODE-840617
