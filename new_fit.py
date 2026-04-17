# %%
import numpy as np
from matplotlib import pyplot as plt
import csv

# %matplotlib inline

import xarray as xr
import lib_ecofun as lef
from importlib import reload
reload(lef)
from scipy.optimize import curve_fit, minimize, dual_annealing, basinhopping, brute, differential_evolution, shgo#, direct

import pickle
# %%
lef.beta_fun(0, 1, delta_sig= 0.5)

# %% [markdown]
# # Model tuning

# %% [markdown]
# ### Setting inicond and obs to fit

# %%
fcu = 0.8
year_ini = 2000
inicond = lef.inicond_yr(year_ini)
#inicond['Kf_ini'] = inicond['Kf_ini']*lef.fossil_capacity_util/fcu

obs = dict()
obs['Ig_ratio'] = lef.Ig_obs/(lef.Ig_obs+lef.If_obs) # considering only investment in power generation capacity (no grids, storage, EVs, ...)

# E_obs = xr.load_dataarray('Etot_hist_1965-2022.nc')
# E_obs /= E_obs.sel(year = 2000)
obs['E'] = lef.final_energy_yr(year_ini)

# obs['Eg_ratio'] = (lef.Eg_ratio_fe * (obs['E']) - 0.15)/obs['E']
obs['Eg_ratio'] = lef.Eg_ratio_fe

# %% [markdown]
# ### Chose params to fit and bounds (initial guess only needed for minimize)

# %%
# parnames = ['growth', 'beta_0', 'r_inv', 'a', 'b', 'eta_g', 'eta_f']
# bounds = [(0.01, 0.03), (-0.5, 0.5), (0.2, 0.8), (0.5, 1.5), (0.1, 1.5), (0.2, 0.9), (0.2, 0.9)]

# parnames = ['growth', 'beta_0', 'r_inv', 'gamma_g', 'gamma_f', 'eta_g', 'eta_f']
# bounds = [(0.01, 0.03), (-0.5, 0.5), (0.2, 0.9), (0.5, 1.5), (0.1, 1.5), (0.2, 0.9), (0.2, 0.9)]

# parnames = ['growth', 'beta_0', 'r_inv', 'a', 'b', 'gamma_g', 'eta_g', 'eta_f']
# bounds = [(0.01, 0.03), (-0.5, 0.), (0.2, 0.9), (0.2, 1.), (0.2, 1.), (0.1, 3.), (0.2, 0.9), (0.2, 0.9)]
parnames = ['growth', 'beta_0', 'r_inv', 'gamma_g', 'gamma_f', 'eta_g', 'eta_f']
bounds = [(0.01, 0.03), (-0.5, 0.5), (0.2, 0.9), (0.1, 3.), (0.1, 3.), (0.2, 0.9), (0.2, 0.9)]

initial_guess = [lef.best_params[par] for par in parnames]

param_bounds = {par: boun for par, boun in zip(parnames, bounds)}

# parnames = ['growth', 'beta_0', 'r_inv', 'a', 'delta_sig']
# bounds = [(0.02, 0.03), (-0.3, 0.3), (0.2, 0.6), (0.5, 1.5), (0.3, 2.)]
# initial_guess = [0.02, 0., 0.1, 1., 1., 0.7, 0.7]

# %%
initial_guess

# %% [markdown]
# ### Set other options: use public inv? what share? what scenario?

# %%

params = lef.default_params.copy()
print(params)
print('-------------')
params['delta_sig'] = 0.5
params['a'] = 0.76
params['b'] = 1.07
# params['gamma_g'] = 1.
# params['gamma_f'] = 1.
# params['eta_g'] = 0.8
# params['eta_f'] =
#params['growth'] = 0.029 # fixing Growth!

verbose = False

public_investment = False

params['r_inv_state'] = 0.015
arr = np.concatenate([np.linspace(0., 0.7, 2020-year_ini), np.linspace(0.7, 0.7, 80)])
mu_scen = xr.DataArray(arr, dims = ('year'), coords = {'year': np.arange(year_ini, 2100)})

obs_weights = {'Eg_ratio': 10, 'E': 1, 'Ig_ratio': 1}

# %%
params

# %% [markdown]
# ### Fit 1: only market

# %%
from time import time

# %%
time()-time()

# %%
# def cost_function(parset, parnames = ['beta_0', 'gamma_g', 'growth', 'delta_sig'], params = default_params.copy(), year_ini = 2015, inicond = inicond_2015, verbose = False, obs = None, public_investment = False, mu_state_scenario = None, linear_gdp = None, obs_weights = None, break_on_scarcity = False)

result_dict = {}

# tips = 'hop dual diffev shgo direct brute'.split()
# #cose = [basinhopping, dual_annealing, differential_evolution, shgo, direct, brute]

# for tip, cos in zip(tips, cose):
#     print(tip, time())
    
#     result = cos(lef.cost_function, bounds = bounds, args = (parnames, params, year_ini, inicond, verbose, obs, public_investment, mu_scen, obs_weights))
#     print(result.x)


# %%
threshold = 0.05

model_args = (parnames, params, year_ini, inicond, verbose, obs, public_investment, mu_scen, None, obs_weights)

# Collect all evaluations below threshold
below_threshold = []

def callback_wrapper(xk, threshold, args = model_args):
    cost = lef.cost_function(xk, *args)
    if cost < threshold:
        below_threshold.append((xk.copy(), cost))
    return False

# run diffevo with callback
result = differential_evolution(lef.cost_function, bounds, args = model_args, maxiter = 10000, popsize = 100, callback = callback_wrapper)
print(f'AAAAAAAAAAAAAAA diffevo: {result.fun:5.2f}  ', result.x)


with open('popoulation_IgEgE_thres05_finalenergy.p', 'wb') as fi:
    pickle.dump(below_threshold, fi)

# result = dual_annealing(lef.cost_function, bounds, args = model_args)
# print(f'AAAAAAAAAAAAAAA dual: {result.fun:5.2f}  ', result.x)

# model_args = (parnames, params, year_ini, inicond, verbose, obs, public_investment, mu_scen, None, obs_weights, param_bounds)
# result = basinhopping(lef.cost_function, x0= initial_guess, minimizer_kwargs = {'args': model_args}, niter = 1000)
# print(f'AAAAAAAAAAAAAAA hopping: {result.fun:5.2f}  ', result.x)
# # result_dict['hop'] = result


# # %%
# result.x

# # %%
# year_ini = 2000

# params_ok = params.copy()
# params_fit = result.x
# for par, parval in zip(parnames, params_fit):
#     params[par] = parval

# resu_om = lef.run_model(inicond = inicond, params = params_ok, n_iter = 100, verbose = True, rule = 'maxgreen', year_ini = year_ini, public_investment=public_investment, mu_state_scenario=mu_scen)

# # %%
# lef.costfun(resu_om, obs), lef.costfun(resu_best, obs)

# # %%
# best_params = lef.best_params.copy()

# # %%
# best_params

# # %%
# resu_best = lef.run_model(inicond = inicond, params = best_params, n_iter = 100, verbose = True, rule = 'maxgreen', year_ini = year_ini, public_investment=public_investment, mu_state_scenario=mu_scen)

# # %%


# # %%
# figs_obs = lef.plot_resuvsobs_ds(resu_om, obs)

# # %%
# lef.plot_resu(resu_om, year_ini = year_ini)


