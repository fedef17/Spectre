# %%
# %%
import numpy as np
from matplotlib import pyplot as plt
import csv

# %matplotlib inline

import xarray as xr
import lib_ecofun as lef

from scipy.optimize import curve_fit, minimize, dual_annealing, basinhopping, brute, differential_evolution, shgo


# %%
from importlib import reload
import os
import sys

reload(lef)

gdp2 = lef.read_gdp_owid()

#tag = 'gdp_noinfl_1804/'
#ok_gdp = gdp2.sel(year = slice(2000, None))
#gdp_fit = ok_gdp # xr.DataArray(spezzata, coords={"year": ok_gdp.year}, dims="year")

tag = 'gdp_current_1804/'
ok_gdp = lef.gdp.sel(year = slice(2000, None))
gdp_fit = ok_gdp # xr.DataArray(spezzata, coords={"year": ok_gdp.year}, dims="year")

do_dualfit = False
do_diffevofit = False
###############################################################################################

# %%
cart_figs = tag
if not os.path.exists(cart_figs): os.mkdir(cart_figs)

# %%
rainbow_palette_5 = [
    "#D70000",  # Dark Red
    "#FFC700",  # Gold
    "#008700",  # Dark Green
    "#0057A0",  # Dark Blue
    "#7B1FA2"  # deep purple
]

rainbow_palette_10 = [
    "#C0001A",  # deep red
    "#E84B1A",  # vermillion
    "#E87E00",  # burnt orange
    "#C8A800",  # golden yellow
    "#5A9E00",  # olive green
    "#00893E",  # forest green
    "#007A8C",  # teal
    "#0057B8",  # cobalt blue
    "#3D2FBF",  # indigo
    "#7B1FA2",  # deep purple
]

rainbow_palette_9 = [
    "#C0001A",  # deep red
    "#E84B1A",  # vermillion
    "#E87E00",  # burnt orange
    "#C8A800",  # golden yellow
    "#00893E",  # forest green
    "#007A8C",  # teal
    "#0057B8",  # cobalt blue
    "#3D2FBF",  # indigo
    "#7B1FA2",  # deep purple
]

###############################################33

# %%
allobs = lef.define_obs(None, adimensional=True)
allobs_dim = lef.define_obs(None, adimensional=False)

year_ini = 2000
params = lef.default_params.copy()
verbose = False
public_investment = False
params['r_inv_state'] = 0.015
arr = np.concatenate([np.linspace(0., 0.7, 20), np.linspace(0.7, 0.7, 80)])
mu_scen = xr.DataArray(arr, dims = ('year'), coords = {'year': np.arange(year_ini, year_ini + 100)})

obs_weights = None

result_dict = {}

# %%
params['a'] = 2.3
params['b'] = 1.4

# %%
params_prime = params.copy()

params_prime['a'] = lef.a_prime(year_ini, a = params['a'])
params_prime['b'] = lef.a_prime(year_ini, a = params['b'])

params_prime_range = {
 'growth': (0.01, 0.05),
 'delta_sig': (0.2, 1.),
 'a': (0.6, 1.1),
 'b': (0.3, 0.8),
 'gamma_f': (0.1, 0.7),
 'gamma_g': (0.1, 0.7),
 'eta_g': (0.1, 0.95),
 'eta_f': (0.1, 0.95),
 'r_inv': (0.05, 0.5),
 'beta_0': (-0.5, 0.5)}

# %%
inicond = lef.inicond_yr(year_ini, params_prime, adimensional = True, fcu = lef.fossil_capacity_util)

# %%
gdp_scen = lef.read_gdp_scenarios()

fig = plt.figure(figsize = (12,8))
all_scen = []
gdp2corr = gdp2*(gdp_scen['SSP1'].sel(year = 2005)/gdp2.sel(year = 2005))

for ssp, col in zip(['SSP1', 'SSP2', 'SSP3', 'SSP4', 'SSP5'], rainbow_palette_5):
    mismatch = gdp_scen[ssp].sel(year = 2024) - gdp_fit.sel(year = 2024)
    misfact = gdp_scen[ssp].sel(year = 2024)/gdp_fit.sel(year = 2024)
    misfact2 = gdp_scen[ssp].sel(year = 2024)/gdp2corr.sel(year = 2024)

    gdp_scen[ssp].plot(label = ssp + ' (orig)', color = col)

    fitto = xr.concat([gdp_fit, gdp_scen[ssp].sel(year = slice(2025, None))-mismatch.values], dim = 'year')
    fitto.plot(label = ssp + ' (shifted)', color = col, ls = '--')

    fitto2 = xr.concat([gdp_fit, gdp_scen[ssp].sel(year = slice(2025, None))/misfact.values], dim = 'year')
    fitto2.plot(label = ssp + ' (scaled 1)', color = col, ls = ':')

    fitto3 = xr.concat([gdp2corr, gdp_scen[ssp].sel(year = slice(2025, None))/misfact2.values], dim = 'year')
    fitto3.plot(label = ssp + ' (scaled 2)', color = col, ls = '-.')

    all_scen.append(fitto2/fitto2.sel(year = 2000)) # Using fitto2

allobs_dim['Y'].sel(year = slice(2000, None)).plot.scatter(color = 'black', s = 70, label = 'obs')
gdp2corr.sel(year = slice(2000, None)).plot.scatter(color = 'black', s = 70, marker = 'x', label = 'obs (no infl., scaled)')
plt.legend()
plt.ylabel('GDP (billion US dollars)')
plt.xlim(2000, 2050)
plt.ylim(0, 0.4e6)

fig.savefig(cart_figs + 'GDP_SSP_scaling.pdf')

# %%
fig = plt.figure(figsize = (12,8))
gdp2corr = gdp2*(gdp_scen['SSP1'].sel(year = 2005)/gdp2.sel(year = 2005))

for ssp, col in zip(['SSP1', 'SSP2', 'SSP3', 'SSP4', 'SSP5'], rainbow_palette_5[::-1]):
    mismatch = gdp_scen[ssp].sel(year = 2024) - gdp_fit.sel(year = 2024)
    misfact = gdp_scen[ssp].sel(year = 2024)/gdp_fit.sel(year = 2024)
    misfact2 = gdp_scen[ssp].sel(year = 2024)/gdp2corr.sel(year = 2024)

    gdp_scen[ssp].plot(label = ssp + ' (orig)', color = col, ls = ':')

    fitto = xr.concat([gdp_fit, gdp_scen[ssp].sel(year = slice(2025, None))-mismatch.values], dim = 'year')
    # fitto.plot(label = ssp + ' (shifted)', color = col, ls = '--')

    fitto2 = xr.concat([gdp_fit, gdp_scen[ssp].sel(year = slice(2025, None))/misfact.values], dim = 'year')
    fitto2.plot(label = ssp + ' (scaled)', color = col, ls = '-')

    fitto3 = xr.concat([gdp2corr, gdp_scen[ssp].sel(year = slice(2025, None))/misfact2.values], dim = 'year')
    # fitto3.plot(label = ssp + ' (scaled 2)', color = col, ls = '-.')

allobs_dim['Y'].sel(year = slice(2000, None)).plot.scatter(color = 'black', s = 70, label = 'obs')
# gdp2corr.sel(year = slice(2000, None)).plot.scatter(color = 'black', s = 70, marker = 'x', label = 'obs (no infl., scaled)')
plt.legend()
plt.ylabel('GDP (billion US dollars)')
plt.xlim(2000, 2050)
plt.ylim(0, 0.4e6)

fig.savefig(cart_figs + 'GDP_SSP_scaling_ok.pdf')

obs2 = lef.define_obs(['E', 'Eg_ratio', 'Ig_ratio'], year_ref = year_ini)
obs_weights2 = {'E': 1, 'Eg_ratio': 10, 'Ig_ratio': 1}

# %%
#parnames = ['growth', 'delta_sig', 'beta_0', 'r_inv', 'a', 'b', 'gamma_g']#, 'eta_g', 'gamma_g']
parnames = ['eps', 'delta_sig', 'beta_0', 'r_inv', 'a', 'b', 'gamma_g', 'eta_g']
same_costs = True
params_prime_range['eps'] = (0.2, 0.6)
bounds = [params_prime_range[par] for par in parnames]

# %%
params_fit = params_prime.copy()
params_fit['eta_g'] = 0.2
params_fit['eta_f'] = 0.2
params_fit['eps'] = 0.33
scale_costs = True
same_price = True
recalc_inicond = True

###############################################################################################
#####
#####               the FIT!
#####
####======================================================================================
from contextlib import contextmanager

@contextmanager
def redirect_output(filename):
    original_stdout = sys.stdout
    with open(filename, 'w') as f:
        sys.stdout = f
        yield
    sys.stdout = original_stdout


if do_dualfit:
    # Collect all evaluations below threshold
    threshold = 0.05
    inicond_recalc = inicond.copy()
    scen = all_scen[1] # SSP2

    # below_threshold = []
    # def callback_wrapper_dual(xk, cost, context, args = None, **kwargs):
    #     if cost < threshold:
    #         below_threshold.append((xk.copy(), cost))
    #     return False

    print("Starting dual annealing...")
    with redirect_output(cart_figs + 'dual_output.log'):
        resu = dual_annealing(lef.cost_function, bounds, args = (parnames, params_fit, year_ini, inicond_recalc, verbose, obs2, obs_weights2, public_investment, mu_scen, same_costs, scale_costs, same_price, recalc_inicond, None, 'custom', scen))#, callback = callback_wrapper_dual)

    print(f'AAAAAAAAAAAAAAA dual: {resu.fun:6.3f}  ', resu.x)

    for par, parval in zip(parnames, resu.x):
        params_fit[par] = parval
        
    if same_price: params_fit['gamma_f'] = params_fit['gamma_g']
    if same_costs: params_fit['eta_f'] = params_fit['eta_g']

    inicond_recalc = lef.inicond_yr(year_ini, params_fit, adimensional = True, fcu = lef.fossil_capacity_util)

    with redirect_output(cart_figs + 'final_results_dual.log'):
        print(f'Best cost: {resu.fun:6.3f}\n')
        print('')
        print('PARAMS:')
        for par in params_fit:
            print(par, params_fit[par])
        
        print('')
        print('INICOND:')
        for co in inicond_recalc:
            print(co, inicond_recalc[co])

# import pickle
# with open(cart_figs + 'popoulation_dual_005.p', 'wb') as fi:
#     pickle.dump(below_threshold, fi)

## Repeat with diffevo.
if do_diffevofit:
    params_fit2 = params_prime.copy()
    params_fit2['eta_g'] = 0.2
    params_fit2['eta_f'] = 0.2
    params_fit2['eps'] = 0.33
    scale_costs = True
    same_price = True
    recalc_inicond = True

    inicond_recalc = inicond.copy()
    scen = all_scen[1] # SSP2

    # below_threshold = []
    # def callback_wrapper_diff(xk, args = None, **kwargs):
    #     cost = lef.cost_function(xk, *args)
    #     if cost < threshold:
    #         below_threshold.append((xk.copy(), cost))
    #     return False
        
    model_args = (parnames, params_fit2, year_ini, inicond_recalc, verbose, obs2, obs_weights2, public_investment, mu_scen, same_costs, scale_costs, same_price, recalc_inicond, None, 'custom', scen)

    print("Starting differential evolution...")
    # run diffevo with callback
    with redirect_output(cart_figs + 'diffevo_output.log'):
        resu = differential_evolution(lef.cost_function, bounds, args = model_args, maxiter = 10000, popsize = 100)#, callback = callback_wrapper_diff)

    print(f'AAAAAAAAAAAAAAA diffevo: {resu.fun:5.2f}  ', resu.x)

    for par, parval in zip(parnames, resu.x):
        params_fit2[par] = parval
        
    if same_price: params_fit2['gamma_f'] = params_fit2['gamma_g']
    if same_costs: params_fit2['eta_f'] = params_fit2['eta_g']

    inicond_recalc2 = lef.inicond_yr(year_ini, params_fit2, adimensional = True, fcu = lef.fossil_capacity_util)

    # with open(cart_figs + 'popoulation_diffevo_005.p', 'wb') as fi:
    #     pickle.dump(below_threshold, fi)

    with redirect_output(cart_figs + 'final_results_diffevo.log'):
        print(f'Best cost: {resu.fun:6.3f}\n')
        print('')
        print('PARAMS:')
        for par in params_fit2:
            print(par, params_fit2[par])
        
        print('')
        print('INICOND:')
        for co in inicond_recalc2:
            print(co, inicond_recalc2[co])

###############################################################################################################################
###############################################################################################################################
#### Read fit result

def parse_parameter_file(filename):
    params_fit = {}
    inicond_recalc = {}
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    current_section = None
    for line in lines:
        line = line.strip()
        
        if line == 'PARAMS:':
            current_section = 'params'
        elif line == 'INICOND:':
            current_section = 'inicond'
        elif line and current_section:
            # Parse key-value pairs (assuming space-separated)
            parts = line.split()
            if len(parts) >= 2:
                key = parts[0]
                try:
                    value = float(parts[1]) if '.' in parts[1] else int(parts[1])
                except ValueError:
                    value = parts[1]  # Keep as string if not numeric
                
                if current_section == 'params':
                    params_fit[key] = value
                elif current_section == 'inicond':
                    inicond_recalc[key] = value
    
    return params_fit, inicond_recalc

if not do_dualfit:
    print('Reading fit result...')
    params_fit, inicond_recalc = parse_parameter_file(cart_figs + 'final_results_dual.log')
    print("Params:", params_fit)
    print("Inicond:", inicond_recalc)

#########################################################################################################################

# %%
resu_allscen = []
for scen in all_scen:
    resuok = lef.run_model(inicond = inicond_recalc, params = params_fit, n_iter = 100, verbose = True, rule = 'maxgreen', year_ini = year_ini, public_investment=public_investment, mu_state_scenario=mu_scen, scale_costs=scale_costs, gdp_type='custom', gdp_scenario=scen)
    resu_allscen.append(resuok)

# %%
figs = lef.plot_resuvsobs_ds(resu_allscen[1], obs2, year_ok = slice(2000, 2024))

# %%
figs[0].savefig(cart_figs + 'resu_hist_Ig_ratio.pdf')
figs[1].savefig(cart_figs + 'resu_hist_Eg_ratio.pdf')
figs[2].savefig(cart_figs + 'resu_hist_E.pdf')


# %%
all_growths = dict()

for gro in np.arange(0., 0.055, 0.005):
    gdp_gro = []
    Y0 = gdp_fit.sel(year = 2024).values + 0.
    for i in range(2102-2025):
        Y1 = lef.GDP(Y0, growth= gro)
        gdp_gro.append(Y1)
        Y0 = Y1 + 0.

    ds = xr.DataArray(gdp_gro, coords={'year': np.arange(2025, 2102)})

    gdp_fin = xr.concat([gdp_fit, ds], dim = 'year')

    all_growths[gro] = gdp_fin

# %%
resu_groscen = []
for gro in np.arange(0., 0.055, 0.005):
    print(gro)
    scen = all_growths[gro]/all_growths[gro].sel(year = 2000)
    resuok = lef.run_model(inicond = inicond_recalc, params = params_fit, n_iter = 101, verbose = True, rule = 'maxgreen', year_ini = year_ini, public_investment=public_investment, mu_state_scenario=mu_scen, scale_costs=scale_costs, gdp_type='custom', gdp_scenario=scen)
    resu_groscen.append(resuok)

# %%
fig = plt.figure(figsize = (12, 8))

do_all = [1, 0, 1, 0, 1, 1, 1, 0, 1]
for gro, re, col, do in zip(np.arange(0., 0.055, 0.005), resu_groscen, rainbow_palette_9[::-1], do_all):
    if do: re['Eg_ratio'].plot(label = f'Growth: {int(1000*gro)/10} %', color = col)

lef.Eg_ratio.sel(year = slice(2000, None)).plot(label = 'obs', color = 'black')

plt.xlabel('year')
plt.ylabel('Share of renewable energy')
plt.legend()
plt.grid()

fig.savefig(cart_figs + 'resu_groscen_Egratio.pdf')

# %%
fig = plt.figure(figsize = (12, 8))

for gro, re, col, do in zip(np.arange(0., 0.055, 0.005), resu_groscen, rainbow_palette_9[::-1], do_all):
    if do: re['Ig_ratio'].plot(label = f'Growth: {int(1000*gro)/10} %', color = col)

obs2['Ig_ratio'].sel(year = slice(2000, None)).plot(label = 'obs', color = 'black')

plt.xlabel('year')
plt.ylabel('Share of green investment')
plt.legend()
plt.grid()

fig.savefig(cart_figs + 'resu_groscen_Igratio.pdf')

# %%
emiss_scen = dict()

# %%
fig = plt.figure(figsize = (12, 8))

for gro, re, col, do in zip(np.arange(0., 0.055, 0.005), resu_groscen, rainbow_palette_9[::-1], do_all):
    if do: 
        lef.to_emissions(re['Ef']).plot(label = f'Growth: {int(1000*gro)/10} %', color = col)
        emiss_scen[gro] = lef.to_emissions(re['Ef'])

# lef.to_emissions(resu_hist['Ef']).sel(year = slice(2000, 2023)).plot(color = 'grey')
lef.co2.sel(year = slice(2000, None)).plot(color = 'black')
# lef.Eg_ratio.plot(label = 'obs', color = 'black')
plt.ylim(0., 120.)

plt.xlabel('year')
plt.ylabel('CO2 emissions (Gt/year)')
plt.legend()
plt.grid()

fig.savefig(cart_figs + 'resu_groscen_CO2emiss.pdf')



# %%
fig = plt.figure(figsize = (12, 8))

for ssp, re, col in zip([f'SSP{i}' for i in range(1,6)], resu_allscen, rainbow_palette_5[::-1]):
    re['Eg_ratio'].plot(label = "Y from " + ssp, color = col)

lef.Eg_ratio.sel(year = slice(2000, None)).plot(label = 'obs', color = 'black')

plt.xlabel('year')
plt.ylabel('Share of renewable energy')
plt.legend()
plt.grid()
fig.savefig(cart_figs + 'resu_sspscen_Egratio.pdf')

# %%
fig = plt.figure(figsize = (12, 8))

for ssp, re, col in zip([f'SSP{i}' for i in range(1,6)], resu_allscen, rainbow_palette_5[::-1]):
    lef.to_emissions(re['Ef']).plot(label = "Y from " + ssp, color = col)
    emiss_scen[ssp] = lef.to_emissions(re['Ef'])

# lef.to_emissions(resu_hist['Ef']).sel(year = slice(2000, 2023)).plot(color = 'grey')
lef.co2.sel(year = slice(2000, None)).plot(color = 'black')
# lef.Eg_ratio.plot(label = 'obs', color = 'black')

plt.xlabel('year')
plt.ylabel('CO2 emissions (Gt/year)')
plt.legend()
plt.grid()
fig.savefig(cart_figs + 'resu_sspscen_CO2emiss.pdf')

# %%
fig = plt.figure(figsize = (12, 8))

for ssp, re, col in zip([f'SSP{i}' for i in range(1,6)], resu_allscen, rainbow_palette_5[::-1]):
    re['Ig_ratio'].plot(label = "Y from " + ssp, color = col)

obs2['Ig_ratio'].sel(year = slice(2000, None)).plot(label = 'obs', color = 'black')

plt.xlabel('year')
plt.ylabel('Share of green investment')
plt.legend()
plt.grid()
fig.savefig(cart_figs + 'resu_sspscen_Igratio.pdf')

# %% [markdown]
# ## Example of scenario with 2% growth

# %%
figs = lef.plot_resu(resu_groscen[4])
figs[0].savefig(cart_figs + 'resu_gro2_K.pdf')
figs[1].savefig(cart_figs + 'resu_gro2_E.pdf')

# %%
import pickle

# %%
with open('results_neweq1_020426.p', 'wb') as filo:
    pickle.dump([resu_groscen, resu_allscen], filo)

# %%
allobs['E']

# %%
[resu_groscen, resu_allscen] = pickle.load(open('results_neweq1_020426.p', 'rb'))

# %%
figs = lef.plot_resu(resu_groscen[6])
figs[0].savefig(cart_figs + 'resu_gro3_K.pdf')
figs[1].savefig(cart_figs + 'resu_gro3_E.pdf')

# %%
figs = lef.plot_resu(resu_groscen[5])
figs[0].savefig(cart_figs + 'resu_gro25_K.pdf')
figs[1].savefig(cart_figs + 'resu_gro25_E.pdf')

# %%
fig, axs = plt.subplots(1, 3, figsize = (16, 5))

for i, (ax, resu, tit) in enumerate(zip(axs, resu_groscen[4:7], [f'Growth: {int(1000*gro)/10} %' for gro in [0.02, 0.025, 0.03]])):
    ax.fill_betweenx(np.arange(0, 10, 1), 2000, 2024, color = 'lightgray', alpha = 0.5)
    allobs['E'].sel(year = slice(2000, None)).plot.scatter(ax = ax, color = 'blue', s = 5, alpha = 0.5)
    allobs['Ef'].sel(year = slice(2000, None)).plot.scatter(ax = ax, color = 'orange', s = 5, alpha = 0.5)
    allobs['Eg'].sel(year = slice(2000, None)).plot.scatter(ax = ax, color = 'green', s = 5, alpha = 0.5)

    resu['E'].plot(ax = ax, label = 'Total')
    resu['Ef'].plot(ax = ax, label = 'Fossil')
    resu['Eg'].plot(ax = ax, label = 'Green')

    if not np.isnan(resu.year_peak):
        ax.axvline(resu.year_peak, color = 'indianred', lw = 0.5, ls = ':')
    if not np.isnan(resu.year_halved):
        ax.axvline(resu.year_halved, color = 'grey', lw = 0.5, ls = ':')
    if not np.isnan(resu.year_zero):
        ax.axvline(resu.year_zero, color = 'forestgreen', lw = 0.5, ls = ':')

    ax.set_title(tit)
    ax.set_xlabel('year')
    if i == 0:
        ax.set_ylabel('Energy production')
    else:
        ax.set_ylabel('')
        ax.set_yticks([])

    if i == 0: ax.legend()

ylim = ax.get_ylim()
for ax in axs[:-1]:
    ax.set_ylim(ylim)

fig.savefig(cart_figs + 'energy_prod_growth2-3.pdf')

# %%
fig, axs = plt.subplots(1, 2, figsize = (12, 5))

for i, (ax, resu, tit) in enumerate(zip(axs, [resu_groscen[2], resu_allscen[0]], ['Growth: 1%', 'Y from SSP1'])):
    ax.fill_betweenx(np.arange(0, 6, 1), 2000, 2024, color = 'lightgray', alpha = 0.5)
    allobs['E'].sel(year = slice(2000, None)).plot.scatter(ax = ax, color = 'blue', s = 5, alpha = 0.5)
    allobs['Ef'].sel(year = slice(2000, None)).plot.scatter(ax = ax, color = 'orange', s = 5, alpha = 0.5)
    allobs['Eg'].sel(year = slice(2000, None)).plot.scatter(ax = ax, color = 'green', s = 5, alpha = 0.5)

    resu['E'].plot(ax = ax, label = 'Total')
    resu['Ef'].plot(ax = ax, label = 'Fossil')
    resu['Eg'].plot(ax = ax, label = 'Green')

    if not np.isnan(resu.year_peak):
        ax.axvline(resu.year_peak, color = 'indianred', lw = 0.5, ls = ':')
    if not np.isnan(resu.year_halved):
        ax.axvline(resu.year_halved, color = 'grey', lw = 0.5, ls = ':')
    if not np.isnan(resu.year_zero):
        ax.axvline(resu.year_zero, color = 'forestgreen', lw = 0.5, ls = ':')

    ax.set_title(tit)
    ax.set_xlabel('year')
    if i == 0:
        ax.set_ylabel('Energy production')
    else:
        ax.set_ylabel('')
        ax.set_yticks([])

    if i == 0: ax.legend()

    ax.set_xlim((1995, 2100))

ylim = ax.get_ylim()
for ax in axs[:-1]:
    ax.set_ylim(ylim)

ax.set_xlim((1995, 2100))

fig.savefig(cart_figs + 'energy_prod_gro1_vs_SSP1.pdf')

# %%
fig, ax = plt.subplots(figsize = (8, 5))

resu = resu_allscen[0]
tit = 'Y from SSP1'

ax.fill_betweenx(np.arange(0, 6, 1), 2000, 2024, color = 'lightgray', alpha = 0.5)
allobs['E'].sel(year = slice(2000, None)).plot.scatter(ax = ax, color = 'blue', s = 5, alpha = 0.5)
allobs['Ef'].sel(year = slice(2000, None)).plot.scatter(ax = ax, color = 'orange', s = 5, alpha = 0.5)
allobs['Eg'].sel(year = slice(2000, None)).plot.scatter(ax = ax, color = 'green', s = 5, alpha = 0.5)

resu['E'].plot(ax = ax, label = 'Total')
resu['Ef'].plot(ax = ax, label = 'Fossil')
resu['Eg'].plot(ax = ax, label = 'Green')

if not np.isnan(resu.year_peak):
    ax.axvline(resu.year_peak, color = 'indianred', lw = 0.5, ls = ':')
if not np.isnan(resu.year_halved):
    ax.axvline(resu.year_halved, color = 'grey', lw = 0.5, ls = ':')
if not np.isnan(resu.year_zero):
    ax.axvline(resu.year_zero, color = 'forestgreen', lw = 0.5, ls = ':')

ax.set_title(tit)
ax.set_xlabel('year')
ax.set_ylabel('Energy production')

if i == 0: ax.legend()

fig.savefig(cart_figs + 'energy_prod_SSP1.pdf')

# %% [markdown]
# ## Now: convert to temperature scenarios

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import os

from fair import FAIR
from fair.interface import fill, initialise
from fair.io import read_properties

import random

# %%
import fair
fair.__version__

# %% [markdown]
# ### Create model

# %%
def extend_dataarray_to_2100(da, extend_to_year = 2100, time_dim = 'year'):
    """Extend a yearly DataArray to 2100 by repeating the last year's values."""
    last_year_data = da.isel({time_dim: -1})
    last_year = int(da[time_dim].values[-1])
    
    new_years = np.arange(last_year + 1, extend_to_year + 1)
    repeated = xr.concat([last_year_data] * len(new_years), dim=time_dim)
    repeated[time_dim] = new_years

    return xr.concat([da, repeated], dim=time_dim)

# %%
emiss_scen_ok = dict()
for ke in emiss_scen:
    print(ke)
    if ke not in [0.045]:
        if isinstance(ke, float):
            nuke = f'gro{int(ke*1000)/10}'
        else:
            nuke = ke
        if emiss_scen[ke].year[-1] < 2100: 
            ds = extend_dataarray_to_2100(emiss_scen[ke])
        else:
            ds = emiss_scen[ke] 
        emiss_scen_ok[nuke] = ds.sel(year = slice(2025, None))/ds.sel(year = 2025)


# %%
# Define number of ensemble members
n_ens = 20
ok_config = 'central' #f.define_configs(['high', 'central', 'low'])
allscen = list(emiss_scen_ok.keys()) # This has to be a list!!
ens_names = ['ens{:03d}'.format(i) for i in range(n_ens)]

y_end = 2100
y_ini = 2025

f = FAIR()
f.define_time(1750, y_end, 1)
f.define_scenarios(allscen)
f.define_configs(['ens{:03d}'.format(i) for i in range(n_ens)])

# Select only basic species
species, properties = read_properties('fair_data/species_configs_properties.csv')
properties['CH4']['input_mode'] = 'emissions' # Set ch4 and n2o in emission mode
properties['N2O']['input_mode'] = 'emissions'
f.define_species(species, properties)

# For all species run:
# species_all, properties_all = read_properties()

# Defines the model
f.allocate()

f.fill_species_configs('fair_data/species_configs_properties.csv')

# %% [markdown]
# ### Select parameters and create ensemble

# %%
# Step 1: Read the existing CSV file
input_file = 'fair_data/basic_run_example/configs_ensemble.csv'  # Specify your input file name
df = pd.read_csv(input_file)

# Step 2: Extract the central sensitivity setup (high, central, low)
params = df[df.iloc[:, 0] == 'central']

random_seeds = np.array([random.randint(1, int(1e7)) for i in range(n_ens)])
new_data = {col: [params[col].values[0]] * n_ens if col != 'seed' else random_seeds for col in df.columns}
new_data['Unnamed: 0'] = ens_names # names
ensemble = pd.DataFrame(new_data)


if not os.path.exists('fair_data/ensemble/'): os.mkdir('fair_data/ensemble/')
output_file = f'fair_data/ensemble/configs_central_n{n_ens}.csv'  # Specify your output file name
ensemble.to_csv(output_file, index=False)

print(f"\nNew data written to {output_file}")


# %%
f.override_defaults(f'fair_data/ensemble/configs_central_n{n_ens}.csv')

# %% [markdown]
# ### Emissions

# %%
emission_all = pd.read_csv('/home/fedef/Research/git/explore-extensions/data/emissions/extensions_1750-2500.csv')

anni_hist = [('{:6.1f}'.format(i+0.5)) for i in range(1750, y_ini-1)]
print(np.unique(emission_all['scenario']))

emissions = dict()
for cos in ['CO2 FFI', 'CO2 AFOLU', 'Sulfur', 'CH4', 'N2O']:
    emissions[cos] = emission_all[(emission_all['variable'] == cos) & (emission_all['scenario'] == 'medium-overshoot')][anni_hist].values.squeeze()
    #print(emissions[cos].shape)

# %%
ifut = y_end - y_ini +1

fut_emiss = emiss_scen_ok

for scen in allscen:
    for cos in ['CO2 FFI', 'CO2 AFOLU', 'Sulfur', 'CH4', 'N2O']:
        # if cos != 'CO2 FFI':
        #     f.emissions.loc[{'specie': cos, 'scenario': scen}] = np.vstack(n_ens * [np.append(emissions[cos], np.zeros(76))]).T
        # else:
        # Now all emissions are proportional to the CO2 FFI
        f.emissions.loc[{'specie': cos, 'scenario': scen}] = np.vstack(n_ens * [np.append(emissions[cos], emissions[cos][-1]*fut_emiss[scen])]).T

f.forcing.loc[{'specie': 'Volcanic'}] = 0.
# f.concentration.loc[{'specie': 'CH4'}] = 1900.
# f.concentration.loc[{'specie': 'N2O'}] = 336.

# %%
# Starting from pre-industrial
from fair.interface import initialise

for cos in ['CO2', 'CH4', 'N2O']:
    initialise(f.concentration, f.species_configs.loc[{'specie': cos}].baseline_concentration[0], specie=cos)

initialise(f.forcing, 0.)
initialise(f.temperature, 0.)
initialise(f.cumulative_emissions, 0.)
initialise(f.ocean_heat_content_change, 0.)

# %%
f.run()

# %%
f.cumulative_emissions.config

# %%
for scen in allscen:
    f.cumulative_emissions.sel(specie = 'CO2 FFI', scenario = scen, config = 'ens000').sel(timebounds = slice(2000, None)).plot()
    plt.legend()

# %%
rainbow_palette = [
    "#D70000",  # Dark Red
    "#E56000",  # Dark Orange
    "#FFC700",  # Gold
    "#008700",  # Dark Green
    "#0057A0",  # Dark Blue
    "#4B0082"   # Indigo (Dark Purple)
]

# %%
#colors = ['steelblue', 'orange', 'forestgreen', 'indianred', 'violet']
colors = rainbow_palette[::-1]

# %%
def plot_scen(scenlist, colors, istart = 2000, iend = y_end, yave = 5, do_rolling = True, alpha = 0.2, y_shade_ini = 2025, labels = None):
    iro = 0
    if do_rolling: iro = yave

    if labels is None: labels = scenlist

    fig, ax = plt.subplots(figsize = (16,9))
    for scen, col, lab in zip(scenlist, colors, labels):
        mean = f.temperature.sel(scenario = scen, layer = 0, timebounds = slice(y_ini-iro, iend+iro)).mean(['config'])
        std = f.temperature.sel(scenario = scen, layer = 0, timebounds = slice(y_ini-iro, iend+iro)).std(['config'])
        if do_rolling:
            mean = mean.rolling(timebounds = yave, min_periods=yave//2, center=True).mean().sel(timebounds = slice(y_ini, iend))
            std = std.rolling(timebounds = yave, min_periods=yave//2, center=True).mean().sel(timebounds = slice(y_ini, iend))

        ax.fill_between(f.temperature.timebounds.sel(timebounds = slice(y_shade_ini, iend)), (mean-std).sel(timebounds = slice(y_shade_ini, iend)), (mean+std).sel(timebounds = slice(y_shade_ini, iend)), color = col, alpha=alpha, edgecolor = 'none')
        ax.plot(f.temperature.timebounds.sel(timebounds = slice(y_ini, iend)), mean, color = col, lw = 3, label = lab)

    mean = f.temperature.sel(layer = 0, timebounds = slice(istart-iro, y_shade_ini+1+iro)).mean(['config', 'scenario'])
    std = f.temperature.sel(layer = 0, timebounds = slice(istart-iro, y_shade_ini+1+iro)).std(['config', 'scenario'])
    if do_rolling:
        mean = mean.rolling(timebounds = yave, min_periods=yave//2, center=True).mean().sel(timebounds = slice(istart, y_shade_ini+1))
        std = std.rolling(timebounds = yave, min_periods=yave//2, center=True).mean().sel(timebounds = slice(istart, y_shade_ini+1))

    col = 'grey'
    ax.fill_between(f.temperature.timebounds.sel(timebounds = slice(istart,y_shade_ini+1)), (mean-std).sel(timebounds = slice(istart,y_shade_ini+1)), (mean+std).sel(timebounds = slice(istart,y_shade_ini+1)), color = col, alpha=alpha, edgecolor = 'none')
    ax.plot(f.temperature.timebounds.sel(timebounds = slice(istart,y_ini+1)), mean.sel(timebounds = slice(istart,y_ini+1)), label = 'hist', color = col, lw = 3)

    ax.axhline(1.5, color = 'red', ls = ':', lw = 0.5)
    #plt.title('Temperature change')
    plt.xlabel('year')
    plt.ylabel('Temperature anomaly (K)')
    plt.legend()
    plt.grid(color='gray', linestyle=':', linewidth=0.5)    

    return fig


colors = [col for col, do in zip(rainbow_palette_9[::-1], do_all) if do]

istart = 2000
iend = 2100
yave = 5
do_rolling = True

labels = [f'Growth: {int(1000*gro)/10} %' for gro in np.arange(0, 0.045, 0.005)[np.where(do_all)]]

fig = plot_scen(allscen[:6], colors, istart, iend, yave, do_rolling, alpha = 0.1, labels = labels)
fig.gca().set_ylim(0, 4.2)

fig.savefig(cart_figs + 'resu_groscen_temperature.pdf')

# %%
istart = 2000
iend = 2100
yave = 5
do_rolling = True

labels = [f"Y from SSP{ssp}" for ssp in range(1, 6)]
fig = plot_scen(allscen[-5:], rainbow_palette_5[::-1], istart, iend, yave, do_rolling, alpha = 0.1, y_shade_ini = 2025, labels = labels)
fig.gca().set_ylim(0, 4.2)

fig.savefig(cart_figs + 'resu_sspscen_temperature.pdf')


# %%
import ast

# %%
def read_cost_params(filepath: str) -> tuple[list, list]:
    """Read a file of Cost/params lines, returning lists of costs and param dicts."""
    costs, params = [], []
    with open(filepath) as f:
        for line in f:
            cost = float(line.split("Cost:")[1].split("params:")[0].strip())
            param_dict = ast.literal_eval(line.split("params:")[1].strip())
            costs.append(cost)
            params.append(param_dict)
    return costs, params

# %%
costs, param_sets = read_cost_params('dual_output.log')
costs2, param_sets2 = read_cost_params('diffevo_output.log')
costs = costs + costs2
param_sets = param_sets + param_sets2

# %%
thres = min(costs)*1.5
parset_low = []
costs_low = []
for co, parset in zip(costs, param_sets):
    if co < thres:
        parset_low.append(parset)
        costs_low.append(co)

# %%
oklow = np.random.choice(np.arange(len(parset_low)), 200)
parset_low_ok = np.array(parset_low)[oklow]

fig, axs = plt.subplots(2, 4, figsize = (8,4))
for par, ax in zip(parnames, axs.flatten()):
    ax.hist([p[par] for p in parset_low])
    ax.set_title(par)

plt.tight_layout()

fig.savefig(cart_figs + 'resu_tuning_paramvar.pdf')

resu_sens_gro = dict()

for gro in np.arange(0, 0.055, 0.005):
    scen = all_growths[gro]/all_growths[gro].sel(year = 2000)
    resu_sens_gro[gro] = []

    for co, parset in zip(costs, parset_low_ok):
        inicond_recalc2 = lef.inicond_yr(year_ini, parset, adimensional = True, fcu = lef.fossil_capacity_util)

        resuok = lef.run_model(inicond = inicond_recalc2, params = parset, n_iter = 101, verbose = True, rule = 'maxgreen', year_ini = year_ini, public_investment=public_investment, mu_state_scenario=mu_scen, scale_costs=scale_costs, gdp_type='custom', gdp_scenario=scen)
        resu_sens_gro[gro].append(resuok)


# %%
fig = plt.figure(figsize = (12, 8))

labels = [f'Growth: {int(1000*gro)/10} %' for gro in np.arange(0, 0.055, 0.005)]
for gro, col, lab, best, do in zip(np.arange(0., 0.055, 0.005), rainbow_palette_9[::-1], labels, resu_groscen, do_all):
    if not do: continue
    allresu = resu_sens_gro[gro]
    all_emiss = []
    for re in allresu:
        emiss = lef.to_emissions(re['Ef'])
        emiss.plot(color = col, lw = 0.15)
        all_emiss.append(emiss)

    #xr.concat(all_emiss, dim = 'member').mean('member').plot(color = col, lw = 2, label = lab)
    lef.to_emissions(best['Ef']).plot(color = col, lw = 2, label = lab, ls = '-')
    lef.to_emissions(best['Ef']).sel(year = slice(2070, 2070)).plot.scatter(color = col, s = 50, marker = '*', alpha = 0.5)
    
    #emiss_scen[gro] = lef.to_emissions(re['Ef'])

# lef.to_emissions(resu_hist['Ef']).sel(year = slice(2000, 2023)).plot(color = 'grey')
lef.co2.sel(year = slice(2000, None)).plot(color = 'black')
# lef.Eg_ratio.plot(label = 'obs', color = 'black')

plt.ylim(-5, 120.)

plt.xlabel('year')
plt.ylabel('CO2 emissions (Gt/year)')
plt.legend()
plt.grid()

fig.savefig(cart_figs + 'resu_groscen_CO2emiss_paramvar.pdf')

# %%
resu_sens_ssp = dict()

for scen, ssp in zip(all_scen, [f'SSP{i+1}' for i in range(5)]):
    resu_sens_ssp[ssp] = []
    for co, parset in zip(costs, parset_low_ok):
        inicond_recalc2 = lef.inicond_yr(year_ini, parset, adimensional = True, fcu = lef.fossil_capacity_util)

        resuok = lef.run_model(inicond = inicond_recalc2, params = parset, n_iter = 100, verbose = True, rule = 'maxgreen', year_ini = year_ini, public_investment=public_investment, mu_state_scenario=mu_scen, scale_costs=scale_costs, gdp_type='custom', gdp_scenario=scen)
        resu_sens_ssp[ssp].append(resuok)

# %%
fig = plt.figure(figsize = (12, 8))

labels = [f"Y from SSP{ssp}" for ssp in range(1, 6)]
for ssp, col, lab, best in zip([f'SSP{i}' for i in range(1,6)], rainbow_palette[::-1], labels, resu_allscen):
    allresu = resu_sens_ssp[ssp]
    for re in allresu:
        lef.to_emissions(re['Ef']).plot(color = col, lw = 0.1)

    lef.to_emissions(best['Ef']).plot(color = col, lw = 2, label = lab)
    lef.to_emissions(best['Ef']).sel(year = slice(2070, 2070)).plot.scatter(color = col, s = 50, marker = '*', alpha = 0.5)

# lef.to_emissions(resu_hist['Ef']).sel(year = slice(2000, 2023)).plot(color = 'grey')
lef.co2.sel(year = slice(2000, None)).plot(color = 'black')
# lef.Eg_ratio.plot(label = 'obs', color = 'black')

plt.ylim(0., 120.)

plt.xlabel('year')
plt.ylabel('CO2 emissions (Gt/year)')
plt.legend()
plt.grid()

fig.savefig(cart_figs + 'resu_sspscen_CO2emiss_paramvar.pdf')


import matplotlib.cm as cm
cma = cm.get_cmap('viridis_r')
costok = np.array(costs_low)[oklow]
costmap = (costok-min(costok))/(max(costok)-min(costok))
cols = list(cma(costmap))

# %%
figs = lef.plot_resuvsobs_ds(resu_sens_gro[0.02], obs2, year_ok = slice(2000, 2024), greystyle=True, colors = cols)
best_hist = resu_groscen[2].sel(year = slice(2000, 2024))
ax = figs[0].gca()
best_hist.Ig_ratio.plot(ax = ax, color = 'orange')
ax = figs[1].gca()
best_hist.Eg_ratio.plot(ax = ax, color = 'orange')
ax = figs[2].gca()
best_hist.E.plot(ax = ax, color = 'orange')

figs[0].savefig(cart_figs + 'resu_hist_Ig_ratio_ens.pdf')
figs[1].savefig(cart_figs + 'resu_hist_Eg_ratio_ens.pdf')
figs[2].savefig(cart_figs + 'resu_hist_E_ens.pdf')



# # %% [markdown]
# # ### Cost/Profit ratio

# %%
fig = plt.figure()
resu = resu_delta_m50[0].sel(year = slice(2000, 2025))
(resu.Cg/resu.Eg).plot(label = 'green')
(resu.Cf/resu.Ef).plot(label = 'fossil')
plt.legend()
plt.title('Production cost per unit energy (energy units)')
fig.savefig(cart_figs + 'cost_energy_ratio.pdf')


# %%
fig = plt.figure()
resu = resu_groscen[4].sel(year = slice(2000, 2050))
(resu.Cg/resu.Eg).plot(label = 'green')
(resu.Cf/resu.Ef).plot(label = 'fossil')
plt.legend()
plt.title('Production cost per unit energy (energy units)')
fig.savefig(cart_figs + 'cost_energy_ratio_gro2.pdf')


# # %%
# resu_delta_m20 = []
# resu_delta_p20 = []

# for gro in np.arange(0, 0.055, 0.005):
#     scen = all_growths[gro]/all_growths[gro].sel(year = 2000)

#     parset = params_fit.copy()
#     parset['delta_g'] = 0.007
#     parset['delta_f'] = 0.007

#     inicond_recalc2 = lef.inicond_yr(year_ini, parset, adimensional = True, fcu = lef.fossil_capacity_util)

#     resuok = lef.run_model(inicond = inicond_recalc2, params = parset, n_iter = 100, verbose = True, rule = 'maxgreen', year_ini = year_ini, public_investment=public_investment, mu_state_scenario=mu_scen, scale_costs=scale_costs, gdp_type='custom', gdp_scenario=scen)
#     resu_delta_m20.append(resuok)

#     parset = params_fit.copy()
#     parset['delta_g'] = 0.013
#     parset['delta_f'] = 0.013

#     inicond_recalc2 = lef.inicond_yr(year_ini, parset, adimensional = True, fcu = lef.fossil_capacity_util)

#     resuok = lef.run_model(inicond = inicond_recalc2, params = parset, n_iter = 100, verbose = True, rule = 'maxgreen', year_ini = year_ini, public_investment=public_investment, mu_state_scenario=mu_scen, scale_costs=scale_costs, gdp_type='custom', gdp_scenario=scen)
#     resu_delta_p20.append(resuok)


# fig, axs = plt.subplots(1, 2, figsize = (15,6))
# # fig = plt.figure(figsize = (12, 8))

# for ax, resus, tit in zip(axs, [resu_delta_m20, resu_delta_p20], ['$\delta_{g,f}$ decreased by 30%', '$\delta_{g,f}$ increased by 30%']):
#     for i, (gro, col) in enumerate(zip(np.arange(0., 0.055, 0.005), rainbow_palette_10[::-1])):
#         re1 = resu_groscen[i]
#         lef.to_emissions(re1['Ef']).plot(ax = ax, color = col, ls = ':')

#         re2 = resus[i]
#         lef.to_emissions(re2['Ef']).plot(ax = ax, color = col)
#         # re3 = resu_delta_p20[i]
#         # lef.to_emissions(re3['Ef']).plot(color = col, ls = ':')
#         #emiss_scen[gro] = lef.to_emissions(re['Ef'])

#     # lef.to_emissions(resu_hist['Ef']).sel(year = slice(2000, 2023)).plot(color = 'grey')
#     lef.co2.sel(year = slice(2000, None)).plot(ax = ax, color = 'black')
#     # lef.Eg_ratio.plot(label = 'obs', color = 'black')

#     ax.set_ylim(0, 120)
#     ax.set_xlabel('year')
#     ax.set_ylabel('CO2 emissions (Gt/year)')
#     ax.set_title(tit)
#     plt.legend()
#     plt.grid()

# # %% [markdown]
# # ## Refitting with the 2 deltas

# # %%
# params_fit_deltam50 = {'growth': 0.01,
#  'eps': 0.2475366681297251,
#  'a': 0.6062349118366656,
#  'b': 0.7997883078082685,
#  'gamma_f': 0.4174390158735987,
#  'gamma_g': 0.4174390158735987,
#  'eta_g': 0.2708651763445069,
#  'eta_f': 0.2708651763445069,
#  'h_g': 0.5,
#  'h_f': 0.5,
#  'r_inv': 0.1656755047253298,
#  'beta_0': 0.05011853260955768,
#  'delta_sig': 0.5414417840701883,
#  'delta_g': 0.005,
#  'delta_f': 0.005,
#  'f_heavy': 0.1,
#  'r_inv_state': 0.015}

# # %%
# params_fit_deltap50 = {'growth': 0.01,
#  'eps': 0.24754051900129914,
#  'a': 0.6,
#  'b': 0.8,
#  'gamma_f': 0.2708407780049921,
#  'gamma_g': 0.2708407780049921,
#  'eta_g': 0.281130870472586,
#  'eta_f': 0.281130870472586,
#  'h_g': 0.5,
#  'h_f': 0.5,
#  'r_inv': 0.3006153400290238,
#  'beta_0': 0.11881665442249888,
#  'delta_sig': 0.5365133295235555,
#  'delta_g': 0.015,
#  'delta_f': 0.015,
#  'f_heavy': 0.1,
#  'r_inv_state': 0.015}

# # %%
# resu_delta_m50 = []
# resu_delta_p50 = []

# for gro in np.arange(0, 0.055, 0.005):
#     scen = all_growths[gro]/all_growths[gro].sel(year = 2000)

#     parset = params_fit_deltam50.copy()
#     inicond_recalc2 = lef.inicond_yr(year_ini, parset, adimensional = True, fcu = lef.fossil_capacity_util)

#     resuok = lef.run_model(inicond = inicond_recalc2, params = parset, n_iter = 100, verbose = True, rule = 'maxgreen', year_ini = year_ini, public_investment=public_investment, mu_state_scenario=mu_scen, scale_costs=scale_costs, gdp_type='custom', gdp_scenario=scen)
#     resu_delta_m50.append(resuok)

#     parset = params_fit_deltap50.copy()
#     inicond_recalc2 = lef.inicond_yr(year_ini, parset, adimensional = True, fcu = lef.fossil_capacity_util)

#     resuok = lef.run_model(inicond = inicond_recalc2, params = parset, n_iter = 100, verbose = True, rule = 'maxgreen', year_ini = year_ini, public_investment=public_investment, mu_state_scenario=mu_scen, scale_costs=scale_costs, gdp_type='custom', gdp_scenario=scen)
#     resu_delta_p50.append(resuok)


# fig, axs = plt.subplots(1, 2, figsize = (15,6))
# # fig = plt.figure(figsize = (12, 8))

# for ax, resus, tit in zip(axs, [resu_delta_m50, resu_delta_p50], ['$\delta_{g,f}$ decreased by 50%', '$\delta_{g,f}$ increased by 50%']):
#     for i, (gro, col) in enumerate(zip(np.arange(0., 0.055, 0.005), rainbow_palette_10[::-1])):
#         re1 = resu_groscen[i]
#         lef.to_emissions(re1['Ef']).plot(ax = ax, color = col, ls = ':')

#         re2 = resus[i]
#         lef.to_emissions(re2['Ef']).plot(ax = ax, color = col)
#         # re3 = resu_delta_p20[i]
#         # lef.to_emissions(re3['Ef']).plot(color = col, ls = ':')
#         #emiss_scen[gro] = lef.to_emissions(re['Ef'])

#     # lef.to_emissions(resu_hist['Ef']).sel(year = slice(2000, 2023)).plot(color = 'grey')
#     lef.co2.sel(year = slice(2000, None)).plot(ax = ax, color = 'black')
#     # lef.Eg_ratio.plot(label = 'obs', color = 'black')

#     ax.set_ylim(0, 120)
#     ax.set_xlabel('year')
#     ax.set_ylabel('CO2 emissions (Gt/year)')
#     ax.set_title(tit)
#     plt.legend()
#     plt.grid()

# fig.savefig(cart_figs + 'deltasens_CO2emiss.pdf')
