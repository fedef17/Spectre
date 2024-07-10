#!/usr/bin/python3

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm

################################################################################################################
######################################## Useful data

########## data on investment from IEA (2015 to 2023). https://www.iea.org/reports/world-energy-investment-2023/overview-and-key-findings

lista = '1074 1319 1132 1105 1129 1114 1137 1109 1225 1066 1259 839 1408 914 1617 1002 1740 1050'.split()
Ig_obs_all = np.array(lista[0::2]).astype(float)
If_obs = np.array(lista[1::2]).astype(float)

# Data on green investment for energy production only (only "Renewable power" in clean energy spending)
Ig_obs = np.array('331 340 351 377 451 494 517 596 659'.split()).astype(float)

#######################

########## E_g/E, from 1965 to 2023 (source ourworldindata: https://ourworldindata.org/renewable-energy)
cose = '6.445519 6.516204 6.423987 6.3901453 6.32996 6.2402315 6.2751184 6.231038 5.98148 6.527657 6.5613737 6.2220235 6.216026 6.4746337 6.5883255 6.8036585 6.9859357 7.1871624 7.3960943 7.3479614 7.309479 7.2850266 7.1429477 7.10847 6.9876184 7.182692 7.301195 7.2864876 7.6539183 7.6321683 7.8718243 7.755703 7.847491 7.890869 7.8530593 7.8158455 7.552836 7.5668545 7.3342075 7.518 7.5638204 7.705343 7.7473364 8.245706 8.564856 8.797048 8.980997 9.414955 9.847355 10.218171 10.504495 10.980251 11.337292 11.743186 12.228147 13.404395 13.469198 14.119935 14.562141'.split()

Eg_ratio = np.array(cose).astype(float)
Eg_ratio_ok = Eg_ratio[-9:]

#################################################################################################################
#################################################################################################################

### the model

def sigmoid(x, delta = 1):
    return 1/(1+np.exp(-x/delta))


def GDP(Y, growth = 0.01, invert_time = False):
    if not invert_time:
        return Y * (1+growth)
    else:
        return Y/(1+growth)


########################### parameters ###########################################################################

default_params = dict()
default_params['growth'] = 0.01 # economic growth
default_params['eps'] = 1 # energy efficiency

default_params['a'] = 1 # Energy production per unit of infrastructure/capital (green)
default_params['b'] = 1 # Energy production per unit of infrastructure/capital (fossil)

default_params['gamma_f'] = 0.5 # Energy price (fossil)
default_params['gamma_g'] = 0.5 # Energy price (green)
default_params['eta_g'] = 0.2 # eta_g*gamma : Costs of energy production (green) [0-1]
default_params['eta_f'] = 0.2 # eta_f*gamma : Costs of energy production (fossil) [0-1]
default_params['h_g'] = 0.5 # Exponent for cost scaling with energy (green) [0-1]
default_params['h_f'] = 0.5 # Exponent for cost scaling with energy (green) [0-1]

default_params['r_inv'] = 0.1 # Fraction of profit that is reinvested in energy infrastructure [0-1]
default_params['beta_0'] = 0.2 # Fraction of infrastructure investments guaranteed for green energy (e.g. subsidies) [0-1]
#default_params['beta_2'] = 0.8 # beta_0 + beta_2 sums to 1
default_params['delta_sig'] = 0.5

default_params['delta_g'] = 0.005 # Depreciation of infrastructure/capital (green)
default_params['delta_f'] = 0.005 # Depreciation of infrastructure/capital (fossil)

default_params['f_heavy'] = 0.1 # Fraction of total production not willing to go green (e.g. military, heavy industry) [0-1]

default_inicond = {'Y_ini' : 1, 'Kg_ini' : 0.1, 'Kf_ini' : 0.9}

########################### parameters ###########################################################################


def forward_step(Y, Kg, Kf, params = default_params, rule = 'maxgreen', verbose = False):
    """
    A single iteration of the model.
    """
    success = 0

    #### params ####
    growth = params['growth']
    eps = params['eps']
    a = params['a']
    b = params['b']
    gamma_g = params['gamma_g']
    gamma_f = params['gamma_f']
    eta_g = params['eta_g']
    eta_f = params['eta_f']
    h_g = params['h_g']
    h_f = params['h_f']
    r_inv = params['r_inv']
    beta_0 = params['beta_0']
    delta_sig = params['delta_sig']
    delta_g = params['delta_g']
    delta_f = params['delta_f']
    f_heavy = params['f_heavy']
    #########

    # Energy and infrastructure
    Eg_max = a * Kg # a = 1
    Ef_max = b * Kf # b time dependent, exog. should decrease to 0

    ## Total production?
    # opt 1: exogenous growing Y, tot energy proportional to Y
    E = eps * Y

    if Eg_max + Ef_max < E: 
        success = 2
        if verbose: print(f'Energy scarcity! {Eg_max} {Ef_max} {E}')
        # raise ValueError(f'Energy scarcity! {Eg_max} {Ef_max} {E}')

    if rule == 'maxgreen':
        Eg = Eg_max
        Ef = E-Eg
        if Eg > E:
            Eg = E
            Ef = 0.
    elif rule == 'proportional':
        Eg = Kg/(Kg+Kf) * E
        Ef = Kf/(Kg+Kf) * E
    elif rule == 'whole_capacity': # This makes Y useless
        Eg = Kg
        Ef = Kf
    elif rule == 'fossil_constraint': # military and heavy industry keep using fossil
        Ef_min = f_heavy * Y
        if E-Ef_min < Eg_max:
            Ef = Ef_min
            Eg = E-Ef_min
        else:
            Eg = Eg_max
            Ef = E-Eg
    
    if E == Eg: 
        if verbose: print('Transition completed!')
        success = 1

    # opt 2: endogenous Y (Dafermos)
    #Y = l * E_max

    ## Profit of energy production
    Pg = gamma_g * (Eg - eta_g * Eg**h_g)
    Pf = gamma_f * (Ef - eta_f * Ef**h_f)
    if Pf < 0.: Pf = gamma_f * (1 - eta_f) * Ef # linearity for small Ef
    if Pg < 0.: Pg = gamma_g * (1 - eta_g) * Eg # linearity for small Eg

    ## Investment in energy production
    beta_2 = 1 - beta_0 # sums to 1
    beta = (beta_0 + beta_2*sigmoid((Pg/Kg - Pf/Kf)/(Pf/Kf), delta = delta_sig)) # fraction of green investment: should be limited between 0 and 1
    if verbose: print(beta, (Pg/Kg - Pf/Kf)/(Pf/Kf), Eg, Ef, Pg, Pf)
    
    Ig = beta * r_inv * (Pg + Pf)
    If = (1-beta) * r_inv * (Pg + Pf)
    if verbose: print(Ig, If)

    ## for next step
    ## Capital/infrastructure
    if verbose and Ig < Kg*delta_g: print(f'Green infrastructure decreasing! {Ig} < {Kg*delta_g}')
    if verbose and If < Kf*delta_f: print(f'Fossil infrastructure decreasing! {If} < {Kf*delta_f}')
    Kg = Ig + Kg * (1-delta_g)
    Kf = If + Kf * (1-delta_f)
    Y = GDP(Y, growth = growth)

    # else: # going backwards
    #     Kg = (Kg - Ig)/(1-delta_g)
    #     Kf = (Kf - If)/(1-delta_f)
    #     Y = GDP(Y, growth = growth, invert_time = True)

    return Y, Kg, Kf, E, Eg, Ef, success


def define_Eg(E, Kg, Kf, a, b, f_heavy, rule = 'maxgreen', verbose = False):
    # Energy and infrastructure
    Eg_max = a * Kg # a = 1
    Ef_max = b * Kf # b time dependent, exog. should decrease to 0

    if Eg_max + Ef_max < E: 
        success = 2
        if verbose: print(f'Energy scarcity! {Eg_max} {Ef_max} {E}')
        # raise ValueError(f'Energy scarcity! {Eg_max} {Ef_max} {E}')

    if rule == 'maxgreen':
        Eg = Eg_max
        Ef = E-Eg
        if Eg > E:
            Eg = E
            Ef = 0.
    elif rule == 'proportional':
        Eg = Kg/(Kg+Kf) * E
        Ef = Kf/(Kg+Kf) * E
    elif rule == 'whole_capacity': # This makes Y useless
        Eg = Kg
        Ef = Kf
    elif rule == 'fossil_constraint': # military and heavy industry keep using fossil
        Ef_min = f_heavy * Y
        if E-Ef_min < Eg_max:
            Ef = Ef_min
            Eg = E-Ef_min
        else:
            Eg = Eg_max
            Ef = E-Eg
    
    return Eg, Ef


def backward_step(Y, Kg, Kf, params = default_params, rule = 'maxgreen', verbose = False):
    """
    A single iteration of the model.
    """
    success = 0

    #### params ####
    growth = params['growth']
    eps = params['eps']
    a = params['a']
    b = params['b']
    gamma_g = params['gamma_g']
    gamma_f = params['gamma_f']
    eta_g = params['eta_g']
    eta_f = params['eta_f']
    h_g = params['h_g']
    h_f = params['h_f']
    r_inv = params['r_inv']
    beta_0 = params['beta_0']
    delta_sig = params['delta_sig']
    delta_g = params['delta_g']
    delta_f = params['delta_f']
    f_heavy = params['f_heavy']
    #########

    ## Total production?
    # opt 1: exogenous growing Y, tot energy proportional to Y
    Y = GDP(Y, growth = growth, invert_time=True)
    E = eps * Y

    # Loop to define K
    max_iter = 20
    ii = 0
    thres = 1e-4
    Kgit = Kg
    Kfit = Kf
    cond = True
    while cond or ii > max_iter:
        if verbose: print('ITeration:', ii)
        Eg, Ef = define_Eg(E, Kgit, Kfit, a, b, f_heavy, rule = rule)

        ## Profit of energy production of previous step
        Pg = gamma_g * (Eg - eta_g * Eg**h_g)
        Pf = gamma_f * (Ef - eta_f * Ef**h_f)
        if Pf < 0.: Pf = gamma_f * (1 - eta_f) * Ef # linearity for small Ef
        if Pg < 0.: Pg = gamma_g * (1 - eta_g) * Eg # linearity for small Eg

        ## Investment in energy production
        beta_2 = 1 - beta_0 # sums to 1
        beta = (beta_0 + beta_2*sigmoid((Pg/Kg - Pf/Kf)/(Pf/Kf), delta = delta_sig)) # fraction of green investment: should be limited between 0 and 1
        #if verbose: print(beta, (Pg/Kg - Pf/Kf)/(Pf/Kf), Eg, Ef, Pg, Pf)
        
        Ig = beta * r_inv * (Pg + Pf)
        If = (1-beta) * r_inv * (Pg + Pf)
        #if verbose: print(Ig, If)

        Kgit_old = Kgit
        Kfit_old = Kfit

        Kgit = (Kg - Ig)/(1-delta_g)
        Kfit = (Kf - If)/(1-delta_f)

        if verbose: print(Kgit, Kgit_old)

        cond = abs((Kgit-Kgit_old)/Kgit) > thres
        ii +=1
    
    Kg = Kgit
    Kf = Kfit
    
    # if E == Eg: 
    #     if verbose: print('Transition completed!')
    #     success = 1

    # ## Profit of energy production of previous step
    # Pg = gamma_g * (Eg - eta_g * Eg**h_g)
    # Pf = gamma_f * (Ef - eta_f * Ef**h_f)
    # if Pf < 0.: Pf = gamma_f * (1 - eta_f) * Ef # linearity for small Ef
    # if Pg < 0.: Pg = gamma_g * (1 - eta_g) * Eg # linearity for small Eg

    # ## Investment in energy production
    # beta_2 = 1 - beta_0 # sums to 1
    # beta = (beta_0 + beta_2*sigmoid((Pg/Kg - Pf/Kf)/(Pf/Kf), delta = delta_sig)) # fraction of green investment: should be limited between 0 and 1
    # if verbose: print(beta, (Pg/Kg - Pf/Kf)/(Pf/Kf), Eg, Ef, Pg, Pf)
    
    # Ig = beta * r_inv * (Pg + Pf)
    # If = (1-beta) * r_inv * (Pg + Pf)
    # if verbose: print(Ig, If)

    # Kg = (Kg - Ig)/(1-delta_g)
    # Kf = (Kf - If)/(1-delta_f)

    return Y, Kg, Kf, E, Eg, Ef, success



def run_model(inicond = default_inicond, params = default_params, n_iter = 100, rule = 'maxgreen', verbose = True, run_backwards = False):
    """

    Runs the model. Returns list of lists of outputs: [Y, Kg, Kf, E, Eg, Ef]  (can be improved!)

    Rules are for energy partition when potential production exceeds demand (see forward_step function).

    """

    Y = inicond['Y_ini']
    Kg = inicond['Kg_ini']
    Kf = inicond['Kf_ini']
    resu = []
    for i in range(n_iter):
        if not run_backwards:
            Y, Kg, Kf, E, Eg, Ef, success = forward_step(Y, Kg, Kf, params = params, verbose = verbose, rule = rule)
        else:
            Y, Kg, Kf, E, Eg, Ef, success = backward_step(Y, Kg, Kf, params = params, verbose = verbose, rule = rule)

        resu.append([Y, Kg, Kf, E, Eg, Ef])
        if success == 0: 
            continue
        elif success == 1:
            if verbose: print(f'Transition completed at time: {i}!')
            break
        elif success == 2:
            if verbose: print(f'Energy scarcity at time: {i}!')
            break
    
    resu = rebuild_resu(resu, run_backwards = run_backwards)
    
    if not run_backwards:
        if success == 1: 
            resu['success'] = True
            resu['year_zero'] = i
            resu['year_peak'] = np.argmax(resu['Ef'])

            for ye in range(resu['year_peak'], len(resu['Ef'])):
                if resu['Ef'][ye] <= resu['Ef'][resu['year_peak']]/2.: break
            resu['year_halved'] = ye
            if verbose: print('Peak fossil: {}'.format(resu['year_peak']))
            if verbose: print('Halved fossil: {}'.format(resu['year_halved']))
        else:
            resu['success'] = False
            resu['year_zero'] = np.nan
            resu['year_peak'] = np.nan
            resu['year_halved'] = np.nan

    return resu


def rebuild_resu(resu, run_backwards = False):
    resu = np.stack(resu)
    Ys = resu[:, 0]
    Kgs = resu[:, 1]
    Kfs = resu[:, 2]
    E = resu[:, 3]
    Eg = resu[:, 4]
    Ef = resu[:, 5]
    if run_backwards:
        Ys = Ys[::-1]
        Kgs = Kgs[::-1]
        Kfs = Kfs[::-1]
        E = E[::-1]
        Eg = Eg[::-1]
        Ef = Ef[::-1]

    ok_resu = dict()
    ok_resu['Y'] = Ys
    ok_resu['Kg'] = Kgs
    ok_resu['Kf'] = Kfs
    ok_resu['E'] = E
    ok_resu['Eg'] = Eg
    ok_resu['Ef'] = Ef

    return ok_resu


def calc_sens_param(param_name, frac_pert = 0.5, var_range = None, inicond = default_inicond, params = default_params, n_iter = 100, n_pert = 5):
    """
    Calculates sensitivity to a single parameter. Computes multiple times the model and returns the trajectories.
    """
    if frac_pert < 0 or frac_pert > 1: raise ValueError('var_range should be between 0 and 1')

    if var_range is None: var_range = [default_params[param_name]*(1-frac_pert), default_params[param_name]*(1+frac_pert)]

    nominal = run_model(inicond = inicond, params = params, n_iter = n_iter, verbose = False)
    
    vals = np.linspace(var_range[0], var_range[1], n_pert)
    
    all_resu = []
    var_params = params.copy()
    for val in vals:
        var_params[param_name] = val
        resu = run_model(inicond = inicond, params = var_params, n_iter = n_iter, verbose = False)

        all_resu.append(resu)

    #plot_resu(resu)
    return vals, nominal, all_resu


def get_colors_from_colormap(n_col, colormap_name='RdBu_r'):
    cmap = cm.get_cmap(colormap_name)
    colors = np.array([cmap(i/(n_col-1)) for i in range(n_col)])
    #print(colors)
    return colors


def plot_sens_param(vals, nominal, all_resu, plot_type = 'tuning'):
    """
    Plots output of calc_sens_param.
    """

    if plot_type == 'dynamics':
        fig = plt.figure()
        resu = nominal
        plt.plot(resu['Kf'] + resu['Kg'], label = 'Total', color = 'violet')
        plt.plot(resu['Kf'], label = 'Fossil', color = 'black')
        plt.plot(resu['Kg'], label = 'Green', color = 'green')

        for resu in all_resu:
            plt.plot(resu['Kf'] + resu['Kg'], color = 'violet', ls = ':', lw = 0.5)
            plt.plot(resu['Kf'], color = 'black', ls = ':', lw = 0.5)
            plt.plot(resu['Kg'], color = 'green', ls = ':', lw = 0.5)

        plt.xlabel('time')
        plt.ylabel('Energy infrastructure')
        plt.legend()

        fig2 = plt.figure()
        resu = nominal
        plt.plot(resu['E'], label = 'Total', color = 'violet')
        plt.plot(resu['Ef'], label = 'Fossil', color = 'black')
        plt.plot(resu['Eg'], label = 'Green', color = 'green')
        for resu in all_resu:
            plt.plot(resu['E'], label = 'Total', color = 'violet', ls = ':', lw = 0.5)
            plt.plot(resu['Ef'], label = 'Fossil', color = 'black', ls = ':', lw = 0.5)
            plt.plot(resu['Eg'], label = 'Green', color = 'green', ls = ':', lw = 0.5)

        plt.xlabel('time')
        plt.ylabel('Energy production')
        plt.legend()
    
    elif plot_type == 'tuning':
        fig = plt.figure()
        resu = nominal
        Ig = np.diff(resu['Kg'])
        If = np.diff(resu['Kf'])

        plt.plot((Ig/(Ig+If))[:20], label = 'model', color = 'black')
        plt.plot(Ig_obs/(Ig_obs+If_obs), label = 'obs', color = 'orange')

        colors = get_colors_from_colormap(len(all_resu))

        for resu, col in zip(all_resu, colors):
            Ig = np.diff(resu['Kg'])
            If = np.diff(resu['Kf'])
            plt.plot((Ig/(Ig+If))[:20], color = col, ls = '--', lw = 1)

            # plt.annotate(f'({x_annotate}, {y_annotate:.2f})', xy=(x_annotate, y_annotate), xytext=(x_annotate + 1, y_annotate - 0.5), arrowprops=dict(facecolor='black', shrink=0.05))

        plt.xlabel('time')
        plt.ylabel('Green share of energy investment (beta)')
        plt.legend()

        fig2 = plt.figure()
        resu = nominal
        plt.plot((100*resu['Eg']/resu['E'])[:20], label = 'model', color = 'black')
        plt.plot(Eg_ratio_ok, label = 'obs', color = 'orange')
        
        for resu, col in zip(all_resu, colors):
            plt.plot((100*resu['Eg']/resu['E'])[:20], color = col, ls = '--', lw = 1)

        plt.xlabel('time')
        plt.ylabel('Share of renewable energy')
        plt.legend()

    fig3 = plt.figure()
    year_zeros = [resu['year_zero'] for resu in all_resu]
    year_peaks = [resu['year_peak'] for resu in all_resu]
    year_halveds = [resu['year_halved'] for resu in all_resu]
    for val, yze, ype, yha, col in zip(vals, year_zeros, year_peaks, year_halveds, colors):
        plt.scatter(val, yze, color = col, marker = 'o')
        plt.scatter(val, ype, color = col, marker = '>')
        plt.scatter(val, yha, color = col, marker = 'x')
    
    plt.xlabel('value')
    plt.ylabel('years')
    plt.legend()

    return fig, fig2, fig3


def plot_resuvsobs(resu, year_ini = 2015, ind_ini = 0, ind_fin = 20):
    """
    Plots outputs vs observed green investment and green energy share.
    """

    fig = plt.figure()
    Ig = np.diff(resu['Kg'])
    If = np.diff(resu['Kf'])

    plt.plot(np.arange(year_ini, year_ini + (ind_fin-ind_ini)), (Ig/(Ig+If))[ind_ini:ind_fin], label = 'model', color = 'black')
    plt.plot(np.arange(2015, 2024), Ig_obs/(Ig_obs+If_obs), label = 'obs', color = 'orange')

    plt.xlabel('time')
    plt.ylabel('Green share of energy investment (beta)')
    plt.legend()

    fig2 = plt.figure()
    plt.plot(np.arange(year_ini, year_ini + (ind_fin-ind_ini)), (100*resu['Eg']/resu['E'])[ind_ini:ind_fin], label = 'model', color = 'black')
    plt.plot(np.arange(2015, 2024), Eg_ratio_ok, label = 'obs', color = 'orange')

    plt.xlabel('time')
    plt.ylabel('Share of renewable energy')
    plt.legend()

    return fig, fig2


def plot_hist(resu, year_ini = 1950, year_fin = 2023):
    """
    Plots outputs vs observed green investment and green energy share.
    """

    fig = plt.figure()
    Ig = np.diff(resu['Kg'])
    If = np.diff(resu['Kf'])

    maxlen = len(If)

    year_ini = max([year_ini, year_fin-maxlen+1])

    plt.plot(np.arange(year_ini, year_fin + 1), (Ig/(Ig+If))[-(year_fin-year_ini+1):], label = 'model', color = 'black')
    plt.plot(np.arange(2015, 2024), Ig_obs/(Ig_obs+If_obs), label = 'obs', color = 'orange')

    plt.xlabel('time')
    plt.ylabel('Green share of energy investment (beta)')
    plt.legend()

    fig2 = plt.figure()
    plt.plot(np.arange(year_ini, year_fin + 1), (100*resu['Eg']/resu['E'])[-(year_fin-year_ini+1):], label = 'model', color = 'black')
    plt.plot(np.arange(1965, 2024), Eg_ratio, label = 'obs', color = 'orange')

    plt.xlabel('time')
    plt.ylabel('Share of renewable energy')
    plt.legend()

    return fig, fig2


def plot_resu(resu):
    fig = plt.figure()
    plt.plot(resu['Kf'] + resu['Kg'], label = 'Total')
    plt.plot(resu['Kf'], label = 'Fossil')
    plt.plot(resu['Kg'], label = 'Green')
    plt.xlabel('time')
    plt.ylabel('Energy infrastructure')
    plt.legend()

    fig2 = plt.figure()
    plt.plot(resu['E'], label = 'Total')
    plt.plot(resu['Ef'], label = 'Fossil')
    plt.plot(resu['Eg'], label = 'Green')
    plt.xlabel('time')
    plt.ylabel('Energy production')
    plt.legend()

    return fig, fig2