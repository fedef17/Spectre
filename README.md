## SPECTRE

SPECTRE, the SimPle Economy-Climate dynamical model for the Transition to Renewable Energy.  SPECTRE is a simplified model of global energy production and investment in a market economy, which aims to give an interpretation of the emergent dynamics in the energy transitions.

### Installation

This is a Python + Jupyter Notebook package.

First, create the env: `mamba env create -f environment.yml`

Then, activate it (`mamba activate spectre`) or select it for your notebook kernel.

### Usage

All functions are in `lib_ecofun.py`. To use them:
```
import lib_ecofun as lef
lef.test()
```

To run a simulation with best-fit parameters and observed initial conditions at `year_ini`:
```
inicond = lef.inicond_yr(year_ini)
params = lef.best_params.copy()
resu_hist = lef.run_model(inicond = inicond, params = params, n_iter = 100, verbose = True, rule = 'maxgreen', year_ini = year_ini)
```

`resu_hist` is a dictionary of xarrays.
