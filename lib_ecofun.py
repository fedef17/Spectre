#!/usr/bin/python3

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import scipy
import xarray as xr
import pandas as pd
import os
import csv

################################################################################################################
######################################## Useful data

datadir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/")

########## data on investment from IEA (2015 to 2023). https://www.iea.org/reports/world-energy-investment-2023/overview-and-key-findings

# The estimates of electricity investment presented in WEI 2023 correspond to
# annual capital spending on new power plants, battery storage and grid assets, or
# the replacement of old assets or refurbishments for life extensions.

# lista = '1074 1319 1132 1105 1129 1114 1137 1109 1225 1066 1259 839 1408 914 1617 1002 1740 1050'.split()
# Ig_obs_all = np.array(lista[0::2]).astype(float)
# If_obs = np.array(lista[1::2]).astype(float)

# # Data on green investment for energy production only (only "Renewable power" in clean energy spending) billion USD
# Ig_obs = np.array('331 340 351 377 451 494 517 596 659'.split()).astype(float)

# Ig_obs = xr.DataArray(Ig_obs, dims = ["year"], coords = {"year": np.arange(2015, 2024)})
# Ig_obs_all = xr.DataArray(Ig_obs_all, dims = ["year"], coords = {"year": np.arange(2015, 2024)})
# If_obs = xr.DataArray(If_obs, dims = ["year"], coords = {"year": np.arange(2015, 2024)})

# Update from WEI 2025
total_investment = xr.load_dataset(datadir + 'total_investment_WEI2025.nc')
Ig_obs = total_investment.green
If_obs = total_investment.fossil
# To remove public from data: (see function read investment data)
# Ig_obs = total_investment['green (corrected)']
# If_obs = total_investment['fossil (corrected)']

co2 = xr.load_dataset('co2_emiss_1750-2024.nc')['co2']

#######################

########## Eg/E, from 1965 to 2024 (source ourworldindata: https://ourworldindata.org/renewable-energy)
cose = [6.088109, 6.155143, 6.0675287, 6.03579, 5.978705, 5.8929, 5.9273896, 
        5.8859487, 5.6497455, 6.168147, 6.201238, 5.8808265, 5.8756404, 
        6.1209283, 6.2307334, 6.4362164, 6.6121817, 6.804729, 7.0058365, 
        6.962811, 6.92851, 6.9085894, 6.774538, 6.7405643, 6.629637, 
        6.795954, 6.917012, 6.902818, 7.2510247, 7.2316523, 7.4576488, 
        7.3469806, 7.43751, 7.482252, 7.4507895, 7.4256287, 7.1826315, 
        7.1862054, 6.975092, 7.1525755, 7.191536, 7.3385572, 7.383397, 
        7.8546634, 8.16082, 8.369888, 8.548672, 8.962558, 9.367213, 
        9.732479, 10.000407, 10.430926, 10.793239, 11.201657, 11.650606, 
        12.780913, 12.861119, 13.503552, 13.942238, 14.822139]

Eg_ratio = np.array(cose).astype(float)/100.
# Eg_ratio_fe.sel(year = slice(2015, 2024)).values = Eg_ratio_fe[-9:]

Eg_ratio = xr.DataArray(Eg_ratio, dims = ["year"], coords = {"year": np.arange(1965, 2025)})

## in TWh from OWID
primary_energy = xr.load_dataarray('Etot_hist_1965-2024.nc') # with substitution method

primary_energy_decomp = xr.load_dataset(datadir + 'primary_energy_substituted_decomp.nc')
## Assuming Etot as sum of green and fossil only
Etot = primary_energy_decomp.green + primary_energy_decomp.fossil

thermal_efficiency_factor = 0.4 # to convert green energy to real primary energy, multiply by thermal efficiency factor

##### Data from: https://www.statista.com/statistics/1325507/oil-and-gas-industry-profits-worldwide/

fossil_profits = np.array([0.11, 0.14, 0.22, 0.91, 0.8 , 0.9 , 0.93, 0.88, 1.66, 1.84, 1.27, 
       0.72, 0.86, 0.85, 0.78, 0.37, 0.55, 0.5 , 0.69, 0.91, 0.47, 0.51, 
       0.52, 0.46, 0.49, 0.64, 0.55, 0.29, 0.46, 0.91, 0.73, 0.68, 0.84, 
       1.17, 1.63, 1.85, 1.92, 2.62, 1.41, 1.84, 2.69, 2.62, 2.43, 2.13, 
       1. , 0.79, 1.11, 1.61, 1.35, 0.87])

Pf_obs = xr.DataArray(fossil_profits, dims = ["year"], coords = {"year": np.arange(1971, 2021)})

######################################################################


#################################################################################################################
#################################################################################################################
default_inicond = {'Y_ini' : 1, 'Kg_ini' : 0.1, 'Kf_ini' : 0.9}

fossil_capacity_util = 0.8 # E/E_max at start; for oil is 0.8 (data from energy institute), but unknown for coal and gas, so likely smaller than 0.8
inicond_2015 = {'Y_ini' : 1, 'Kg_ini' : Eg_ratio.sel(year = 2015).values, 'Kf_ini' : (1-Eg_ratio.sel(year = 2015).values)/fossil_capacity_util} # from 2015
inicond_2000 = {'Y_ini' : 1, 'Kg_ini' :Eg_ratio.sel(year = 2000).values, 'Kf_ini' : (1-Eg_ratio.sel(year = 2000).values)/fossil_capacity_util} # Allowing more fossil capacity at start to avoid scarcity


def define_obs(obs_list = ['E', 'Eg_ratio', 'Ig_ratio'], adimensional = True, year_ref = 2000):
    """
    Defines the observations to fit upon.

    year_ref is the year used to adimensionalize, if adimensional is set to True.
    Usually, that's the initial year.
    """
    allobs = dict()

    allobs['Ig'] = Ig_obs.copy()
    allobs['If'] = If_obs.copy()
    allobs['I'] = allobs['Ig'] + allobs['If']

    allobs['Ig_ratio'] = Ig_obs/(Ig_obs+If_obs)
    allobs['Eg_ratio'] = primary_energy_decomp.green/Etot
    allobs['E'] = Etot.copy()
    allobs['Eg'] = primary_energy_decomp.green.copy()
    allobs['Ef'] = primary_energy_decomp.fossil.copy()

    allobs['Y'] = gdp.copy()

    if adimensional:
        E0 = allobs['E'].sel(year = year_ref).copy()
        Y0 = allobs['Y'].sel(year = year_ref).copy()
        allobs['E'] = allobs['E']/E0
        allobs['Eg'] = allobs['Eg']/E0
        allobs['Ef'] = allobs['Ef']/E0
        allobs['I'] = allobs['I']/Y0
        allobs['Y'] = allobs['Y']/Y0
        allobs['Ig'] = allobs['Ig']/Y0
        allobs['If'] = allobs['If']/Y0

    ## Revenues? Costs?

    # Select only chosen obs
    if obs_list is not None:
        obs = dict()
        for ob in allobs:
            if ob in obs_list:
                obs[ob] = allobs[ob]

        return obs
    else:
        return allobs


def inicond_yr(year, params, adimensional = True, fcu = fossil_capacity_util):
    """
    Defines the initial conditions based on the starting year. Note that Kg_ini and Kf_ini are defined on the basis of the parameter set, since we only have information on the initial energy (K is reconstructed accordingly using the values of a and b).
    """
    #obs = define_obs()
    #     inicond = {'Y_ini' : 1, 'Eg_ini' : Eg_ratio.sel(year = year).values, 'Ef_ini' : (1-Eg_ratio.sel(year = year).values)}
    # else:

    #E0 = primary_energy.sel(year = year).data
    E0 = (primary_energy_decomp.green.sel(year = year).copy() + primary_energy_decomp.fossil.sel(year = year).copy()).data
    Y0 = gdp.sel(year = year).data

    inicond = {'Y_ini': Y0.copy(), 'Eg_ini': primary_energy_decomp.green.sel(year = year).data, 'Ef_ini': primary_energy_decomp.fossil.sel(year = year).data, 'E_ini': E0.copy()}

    Kf0 = define_K_from_E(inicond['Ef_ini'], params['b'], capacity_util = fcu)
    Kg0 = define_K_from_E(inicond['Eg_ini'], params['a'], capacity_util = 1.)

    inicond['Kf_ini'] = Kf0
    inicond['Kg_ini'] = Kg0

    if adimensional:
        inicond['Y_ini'] = inicond['Y_ini']/Y0
        inicond['E_ini'] = inicond['E_ini']/E0
        inicond['Eg_ini'] = inicond['Eg_ini']/E0
        inicond['Ef_ini'] = inicond['Ef_ini']/E0
        # inicond['Kf_ini'] /= Y0
        # inicond['Kg_ini'] /= Y0

        inicond['Kf_ini'] = define_K_from_E(inicond['Ef_ini'], params['b'], capacity_util = fcu)
        inicond['Kg_ini'] = define_K_from_E(inicond['Eg_ini'], params['a'], capacity_util = 1.)

    return inicond


def define_K_from_E(E, a, capacity_util = 1):
    """
    Assumptions:
    - fossil infrastructure is used for a fraction fcu (at start)
    - green infr. is all used
    """
    
    return (E/a)/capacity_util


def define_K_ini(inicond, params, fcu = fossil_capacity_util):
    """
    Assumptions:
    - fossil infrastructure is used for a fraction fcu (at start)
    - green infr. is all used
    """
    inicond['Kf_ini'] = (inicond['Ef_ini']/params['b'])/fcu
    inicond['Kg_ini'] = inicond['Eg_ini']/params['a']

    return inicond


def test():
    print('Library loaded')
    return


def read_energy_data(datadir = datadir, write_nc = False):
    """
    Data on substituted primary energy in TWh (from OWID).
    Substituted means that renewable primary energy is divided by the thermal efficiency factor (0.4) to be comparable with primary energy from fossil fuels.

    Data sources: Energy Institute - Statistical Review of World Energy (2025) Smil (2017) - with major processing by Our World in Data.
    """

    df = pd.read_csv(datadir + 'global-energy-substitution/global-energy-substitution.csv')
    year = df.iloc[:, 2]
    data = df.iloc[:, 3:]

    # Clean column names
    data.columns = [col.split(' (')[0] for col in data.columns]

    # Convert to xarray dataset with year as coordinate
    ds = data.set_index(year.rename('year')).to_xarray()

    # Add total variable
    ds['total'] = sum(ds[var] for var in ds.data_vars)
    ds['fossil'] = sum(ds[var] for var in ['Gas', 'Oil', 'Coal'])
    ds['green'] = sum(ds[var] for var in ['Solar', 'Wind', 'Hydropower', 'Other renewables'])

    if write_nc:
        ds.to_netcdf(datadir + 'primary_energy_substituted_decomp.nc')

    return ds

def read_investment_data(datadir = datadir, write_nc = False):
    # Note: Clean energy includes: Clean Fuels, Direct Air Capture, Transitional fossil fuels, Nuclear, Renewable power, Battery storage, Electricity networks, Fossil fuels: with CCUS, Other clean power and End-use. Other end-use includes: Electrification, renewables for end-use, Hydrogen and industry CCUS. Bunker fuels are only accounted for at the World level.

    # Read Excel file, skip first row
    df = pd.read_excel(datadir + 'WorldEnergyInvestment2025_DataFile.xlsx', sheet_name='World', skiprows=1, header=None)

    # Extract years from second row (index 0 after skiprows)
    #years = np.array(df.iloc[0, 2:].values, dtype = int)
    years = np.array(df.iloc[0, 2:].values).astype(int)

    # Process data starting from third row
    data_dict = {}

    for idx in range(1, len(df)):
        key = df.iloc[idx, 1]
        if pd.isna(key) or 'Note:' in str(key):
            continue
        
        # Determine indent level
        indent = len(key) - len(key.lstrip())
        key = key.strip()
        
        # Clean key: remove units, "of which:", "o/w"
        #key = key.split('(')[0].strip()
        key = key.replace('Billion USD (2024, MER)', '').replace('of which:', '').replace('o/w', '').strip()
        if not key:
            continue
        
        # Get data and create DataArray
        values = np.array(df.iloc[idx, 2:].values).astype(float)
        da = xr.DataArray(values, coords={'year': years}, dims=['year'], name = key)

        data_dict[key] = da
    
    new_dict = {'total': data_dict['Total'], 'fossil': data_dict['Fossil fuels'] + data_dict['Coal (unabated)']+data_dict['Oil and natural gas (unabated)'], 'green': data_dict['Renewables'], 'networks+storage': data_dict['Electricity networks'] + data_dict['Battery storage'], 'end use': data_dict['End-use']}

    total_investment = xr.Dataset(new_dict)

    ##### Add data on public vs private from WEI2025 report.
    # Read from Page 151 of WEI2025 (left plot)
    comm = [2.1, 2, 2, 1.9, 1.9, 1.7, 1.8, 2, 2.25, 2.5, 2.6]
    commercial = 1000*xr.DataArray(comm, coords={'year': years}, dims=['year'])

    public = total_investment.total - commercial
    total_investment['public'] = public
    total_investment['commercial'] = commercial

    ## Considering same page, central plot, in 2024
    # about 30% of fossil investment comes from public
    # about 15% of green investment from public
    # This means that about 50% of the public investment goes to fossil. The other half, is split among energy and grids+end-use. how this is done is unknown.. assuming 25% each.
    # Assuming fraction for 2024 is valid for all years.

    total_investment['fossil (corrected)'] = (1-0.3)*total_investment['fossil']
    total_investment['green (corrected)'] = (1-0.23)*total_investment['green']

    if write_nc:
        total_investment.to_netcdf(datadir + 'total_investment_WEI2025.nc')

    return total_investment


def get_wb_gdp_data(datadir = datadir, finam = "API_NY.GDP.MKTP.CD_DS2_en_csv_v2_207086.csv", last_year = 2024):
#finam = 'API_NY.GDP.MKTP.CD_DS2_en_csv_v2_6298258.csv', last_year = 2022): 
    with open(datadir + finam, newline='') as csvfile:
        reader = csv.reader(csvfile)

        rows = []
        for row in reader:
            rows.append(row)

    rows = rows[5:]

    country = [ro[0] for ro in rows]
    ro_ok = np.where(np.array(country) == 'World')[0][0]
    row_wld = rows[ro_ok]

    gdp = np.array(row_wld[4:], dtype = float)
    years = np.arange(1960, last_year + 1)

    gdp = xr.DataArray(gdp, dims = ["year"], coords = {"year": years})/1e9 # now in billions USD

    return gdp

gdp = get_wb_gdp_data()


def read_gdp_owid(filepath = f"{datadir}/global-gdp-over-the-long-run.csv"):
    """
    Read historical gdp from OWID.
    """

    df = pd.read_csv(filepath, usecols=[2, 3])
    df.columns = ["Year", "GDP"]
    
    ds = xr.DataArray(df['GDP'], coords={'year': df['Year']}, dims="year")

    return ds/1e9



def read_gdp_scenarios(filepath = f"{datadir}/global-gross-domestic-product.csv"):
    """
    Read gdp scenarios from OWID csv file (O'Neill scenarios).
    """

    df = pd.read_csv(filepath, usecols=[0, 1, 2])
    df.columns = ["Entity", "Year", "GDP"]
    
    # Clean entity names
    df["Entity"] = df["Entity"].str.replace(" - Baseline", "", regex=False)
    
    # Pivot to wide format
    df_pivot = df.pivot(index="Year", columns="Entity", values="GDP")
    
    # Reindex to every year and interpolate
    all_years = range(df_pivot.index.min(), df_pivot.index.max() + 1)
    df_pivot = df_pivot.reindex(all_years).interpolate(method="index")
    
    ds = xr.Dataset({col: xr.DataArray(df_pivot[col], dims="year") for col in df_pivot.columns})

    return ds/1e9


### Other data not used in current version
def load_final_energy_data():
    ### Data from IEA Key World Energy Statistics 2021 (https://www.iea.org/data-and-statistics/charts/world-total-final-consumption-by-source-1971-2019)
    ## This is in ExaJoule = 1e18
    # IEA (2021), World total final consumption by source, 1971-2019, IEA, Paris https://www.iea.org/data-and-statistics/charts/world-total-final-consumption-by-source-1971-2019, Licence: CC BY 4.0

    # Data on final energy are not comparable with data on substituted primary energy...
    # something is missing here (substituted primary energy is a simplification anyway, but suited to the low complexity of our model)
    coal=np.array([227,231,235,258,299,345,364,383,402,414,443,464,469,466,470,461,437,421,404,398])
    oil=np.array([1305,1318,1339,1363,1425,1442,1467,1492,1474,1447,1506,1498,1512,1537,1562,1603,1626,1666,1683,1690])
    gas=np.array([469,463,469,485,494,500,508,525,539,516,563,573,572,594,595,595,611,633,671,684])
    biofuels=np.array([367,362,365,372,377,383,392,396,400,400,405,404,408,416,417,419,418,424,428,434])
    electricity=np.array([457,464,408,499,522,545,569,597,607,603,644,664,682,703,721,729,751,775,809,823])
    other=np.array([108,110,109,113,114,114,118,118,116,115,124,130,134,132,131,131,138,142,150,151])

    final_energy = coal + oil + gas + biofuels + electricity + other
    final_energy = xr.DataArray(final_energy, dims = ["year"], coords = {"year": np.arange(2000, 2020)})

    ## Converting to TWh
    final_energy = final_energy*1e18/(1e12*60*60)

    values = [16.87, 16.58, 16.56, 16.39, 16.09, 15.98, 16.01, 15.84, 15.98, 
            16.38, 16.04, 16.01, 16.27, 16.52, 16.66, 16.7, 16.91, 17.1, 
            17.31, 17.69, 19.05, 18.71]

    Eg_ratio_fe = xr.DataArray(np.array(values)/100, dims = ["year"], coords = {"year": np.arange(2000, 2022)})

    return final_energy, Eg_ratio_fe


def get_IRENA_public_inv(filename = '../data/IRENA_Stats_extract_2024_H2.nc'):
    """
    Data on public investment from IRENA 2024
    """
    gigi = xr.load_dataset(filename)/1e3 # Now in billions USD

    # Questo non tornano le unità!
    # Public investment in renewables from IRENA https://www.irena.org/Publications/2024/Jul/Renewable-energy-statistics-2024 
    # yeas = [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
    S_obs = [20024.60, 26487.46, 19466.29, 35188.13, 31186.49, 24778.59, 17403.46, 17447.64, 19630.40, 21676.69]
    S_obs = xr.DataArray(S_obs, dims = ["year"], coords = {"year": np.arange(2013, 2023)})/1e3 # now in billions USD

    return gigi


def get_OWID_IEA_fossil_subs(filename = '../data/fossil_subsidies_owid.nc'):
    """
    Data on public investment in fossil from OWID (IEA)
    """
    gigi = xr.load_dataset(filename) # in billions USD

    return gigi


def get_IISD_green_subs(filename = '../data/IISD_green_support.nc'):
    """
    Data on public investment in green from IISD.
    """
    gigi = xr.load_dataset(filename) # in billions USD

    return gigi

#################################################### parameters ###########################################################################

default_params = dict()
default_params['growth'] = 0.01 # economic growth
default_params['eps'] = 1 # energy efficiency

default_params['a'] = 1 # Energy production per unit of infrastructure/capital (green)
default_params['b'] = 1 # Energy production per unit of infrastructure/capital (fossil)
# default_params['a_linear'] = None # (h, m) a = mx + h

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

default_params['delta_g'] = 0.01 # Depreciation of infrastructure/capital (green)
default_params['delta_f'] = 0.01 # Depreciation of infrastructure/capital (fossil)

default_params['f_heavy'] = 0.1 # Fraction of total production not willing to go green (e.g. military, heavy industry) [0-1]

default_params['r_inv_state'] = 0.01
# default_params['mu_g'] = 1.5
# default_params['mu_f'] = 1
# default_params['delta_sig_state'] = 0.5

##########################################
### Best fit in fit_linearY.ipynb
best_params = default_params.copy()

best_params.update({'growth': 0.0209418049925536, 
                    'beta_0': -0.2831097084724121,
                    'r_inv': 0.11684169484450775,
                    'a': 0.9418414596187906,
                    'delta_sig': 0.47990969261681554
                    })

# Params that give best fit using data of green energy share (2000-2023) and share of green energy investment (2015-2023). Cost function of energy investment is weighted at 0.1 (I_weight). Note delta_sig is at the lowest bound.
best_params_old = {'growth': 0.01877564045416566,
 'eps': 1,
 'a': 1,
 'b': 1,
 'gamma_f': 0.5,
 'gamma_g': 0.5751197750514625,
 'eta_g': 0.2,
 'eta_f': 0.2,
 'h_g': 0.5,
 'h_f': 0.5,
 'r_inv': 0.1,
 'beta_0': -0.1149135946421199,
 'delta_sig': 0.3,
 'delta_g': 0.01,
 'delta_f': 0.01,
 'f_heavy': 0.1}

# Same but with I_weight = 1, which gives slightly worse energy share fit (and faster transition!)
best_params_old_Iw1 = {'growth': 0.017055428532726295,
 'eps': 1,
 'a': 1,
 'b': 1,
 'gamma_f': 0.5,
 'gamma_g': 0.6460875554154768,
 'eta_g': 0.2,
 'eta_f': 0.2,
 'h_g': 0.5,
 'h_f': 0.5,
 'r_inv': 0.1,
 'beta_0': -0.1792655638830066,
 'delta_sig': 0.3,
 'delta_g': 0.01,
 'delta_f': 0.01,
 'f_heavy': 0.1}

###################### Conversion between dimensional and adimensional parameters

def estimate_a_from_investment(timedelta = 2):
    """
    This is to estimate a from investment data (WEI 2025) and energy trends.

    Timedelta is the time for investments to be seen in energy production.

    Assuming timedelta = 2: a = 2.5, b = 1.2
    Assuming timedelta = 1: a = 2.3, b = 1.4
    """

    y0 = 2015
    yf = 2024-timedelta

    deltaKg = Ig_obs.sel(year = slice(y0, yf)).sum()
    deltaEg = primary_energy_decomp.green.sel(year = yf + timedelta) - primary_energy_decomp.green.sel(year = y0 + timedelta)
    a = deltaEg/deltaKg

    deltaKf = If_obs.sel(year = slice(y0, yf)).sum()
    deltaEf = (primary_energy_decomp.fossil.sel(year = yf + timedelta) - primary_energy_decomp.fossil.sel(year = y0 + timedelta))/fossil_capacity_util
    b = deltaEf/deltaKf

    return a.data, b.data


def estimate_a_from_investment_wdelta(delta = 0.01):
    """
    This is to estimate a from investment data (WEI 2025) and energy trends, also consider delta (which was not in previous function).

    Considering only timedelta = 1.
    """
    allobs = define_obs(None)
    allobs_dim = define_obs(None, adimensional = False)

    deltaE = (allobs['Ef'].sel(year = 2024) - allobs['Ef'].sel(year = 2015))
    E0 = allobs['Ef'].sel(year = 2015)
    n = 10
    I_tot = allobs['If'].sel(year = slice(2015, 2024)).sum()
    fut = 0.8

    a_estim = (deltaE * (1 + delta*n/2) + delta*n*E0)/(fut * I_tot)
    a_estim_abs = a_estim * allobs_dim['E'].sel(year = 2000)/allobs_dim['Y'].sel(year = 2000)
    print(f'delta {delta}   ->   adim: {a_estim.values:6.2f} - dim: {a_estim_abs.values:6.2f}')

    deltaE = (allobs['Eg'].sel(year = 2024) - allobs['Eg'].sel(year = 2015))
    E0 = allobs['Eg'].sel(year = 2015)
    n = 10
    I_tot = allobs['Ig'].sel(year = slice(2015, 2024)).sum()
    fut = 1.

    a_estim_g = (deltaE * (1 + delta*n/2) + delta*n*E0)/(fut * I_tot)
    a_estim_g_abs = a_estim_g * allobs_dim['E'].sel(year = 2000)/allobs_dim['Y'].sel(year = 2000)
    print(f'delta {delta}   ->   adim: {a_estim_g.values:6.2f} - dim: {a_estim_g_abs.values:6.2f}')
    
    return a_estim, a_estim_g


def gamma_prime(year, gamma = 0.3):
    """
    To convert energy price to adimensional gamma. Assuming that useful energy is thermal_efficiency_factor * primary/substituted energy.
    
    gamma is in $/kWh, gamma' is adimensional
    """
    E0 = primary_energy.sel(year = year)*1e9*thermal_efficiency_factor
    Y0 = gdp.sel(year = year)*1e9

    return gamma*E0.data/Y0.data


def a_prime(year, a = 1):
    """
    a is in kWh/$, or TWh/billion $. a' is adimensional
    """
    E0 = primary_energy.sel(year = year)
    Y0 = gdp.sel(year = year)

    return a*Y0.data/E0.data

########################### parameters ###########################################################################

### the model

def sigmoid(x, delta = 1):
    return 1/(1+np.exp(-x/delta))


def GDP(Y, growth = 0.01, deltaY = 0, invert_time = False, gdp_type = 'exponential', gdp_scenario = None, year = None, Y_lost = 0):
    """
    Function that determines Y(t+1) given Y(t).

    gdp_type: exponential, linear or custom.
    If custom, function "gdp_function" is used, which takes only year as argument.
    """
    # print('AAAAAAAAAA', Y, deltaY)
    if gdp_type == 'exponential':
        if growth is None: raise ValueError('growth not set for exponential Y')
        if not invert_time:
            Y *= (1+growth)
        else:
            Y /= (1+growth)
    elif gdp_type == 'linear':
        if deltaY is None: raise ValueError('deltaY not set for linear Y')
        Y += deltaY
    elif gdp_type == 'custom':
        if year is None: raise ValueError('year not set for custom Y')
        if gdp_scenario is None: raise ValueError('gdp_scenario not set for custom Y')
        Y = gdp_scenario.sel(year = year)
    
    if Y_lost > 0:
        Y = Y - Y_lost

    return Y

def to_emissions(Ef, CO2_obs = 38.1, Ef_obs = 1.369):
    """
    Convert fossil energy to CO2 emissions. Simple scaling using values for a reference year (2023).
    """
    return CO2_obs*(Ef/Ef_obs)


def plot_cdf_beta(beta_0, prof_ratio, delta_sig):
    x = np.linspace(-3, 3)

    plt.plot(x, cdf(x, sigma = delta_sig))

    return

def cdf(x, mu = 0., sigma = 1.):
    # Compute the integral using the cumulative distribution function (CDF)
    cdf = 0.5 * (1 + scipy.special.erf((x - mu) / (sigma * np.sqrt(2))))
    return cdf


def beta_fun(beta_0, prof_ratio, delta_sig = 1., ftype = 'cdf'):
    """
    Defines fraction of green investment: should be limited between 0 and 1.

    New function cdf assumes gaussian investment around a mean that changes with expected profit and external factors:
        - investment is done randomly and represented by a gaussian distribution
        - the mean value is zero for equal expected profit, or can be different from zero
        - the integral below/above zero gives the two investment ratios

        the dynamics is governed by the ratio between the shift from zero of the mean and the width of the gaussian

        beta_0 is also a displacement
    """
    
    if ftype == 'cdf':
        beta = cdf(0., mu = -(beta_0 + prof_ratio), sigma = delta_sig)
    else:
        beta_2 = 1 - beta_0 # sums to 1
        beta = (beta_0 + beta_2*sigmoid(prof_ratio, delta = delta_sig)) 

    return beta


def prof_ratio(Pg, Pf, Kg, Kf, small = 1e-5):
    """
    Estimates the ratio of profits per unit investment (normalized).
    """
    return (Pg/Kg - Pf/Kf)/(Pg/Kg+Pf/Kf+small)
    #return (Pg/Kg - Pf/Kf)/((Pg+Pf)/(Kg+Kf))

def prod_cost(Eg, eta_g, h_g, etamax, E_crit = 1, scale_costs = True):
    """
    Costs of energy production scale with volume. 
    E_crit = 1 means E_crit = E0 for adimensional, which turns out to work well for the historical fall of green energy production costs.
    """
    Cg = 0

    # func_cost = eta_g * Eg**h_g
    func_cost = eta_g * (Eg/E_crit)**h_g

    if scale_costs:
        if Eg > 0: Cg = min([func_cost, etamax*Eg])
    else:
        Cg = eta_g * Eg

    return Cg


def E_price(Eg, Ef, Kg, Kf, gamma_g, gamma_f, a, b, ref_util_f = 0.8, alpha_util = 1.5, min_price = 0.5, dynamic_price = True, same_price = True):
    """
    Computes the energy price related to capacity utilization.
    alpha_util: elasticity of energy price wrt capacity utilization
    """
    if not dynamic_price:
        return gamma_g, gamma_f
    else:
        gamma_g_dyn = gamma_g # * (Eg/(a * Kg))**alpha_util # useless since green is always at max capacity
        gamma_f_dyn = gamma_f * ((Ef/(b * Kf))/ref_util_f)**alpha_util

        if same_price:
            mean_price = (Eg*gamma_g_dyn + Ef*gamma_f_dyn)/(Eg+Ef)
            gamma_g_dyn, gamma_f_dyn = mean_price, mean_price

        gamma_g_dyn = max(min_price*gamma_g, gamma_g_dyn)
        gamma_f_dyn = max(min_price*gamma_f, gamma_f_dyn)
        
        return gamma_g_dyn, gamma_f_dyn


def forward_step(Y, Kg, Kf, params = default_params, rule = 'maxgreen', betafun_type = 'cdf', verbose = False, raise_bnd_err = False, deltaY = None, mu_state = 0.5, scale_costs = False, public_inv = False, gdp_type = 'exponential', gdp_scenario = None, year_scenario = None, energy_crisis_mode = False, Y_lost = 0, dynamic_price = False, same_price = True):
    """
    Expansion with public investment. Public investment is directed as subsidies, which reduce firms' costs, hence increasing their profits.

    energy_crises_mode: goes on when scarcity is reached, reducing gdp accordingly. 
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

    alpha_util = params['alpha_util'] if 'alpha_util' in params else 1.2
    min_price = params['min_price'] if 'min_price' in params else 0.5
    etamax = 0.9

    ## public inv
    r_inv_state = params['r_inv_state']
    # mu_g = params['mu_g']
    # mu_f = params['mu_f']
    # delta_sig_state = params['delta_sig_state']

    #########
    if verbose: print('params: ', params)

    ## Total production? # opt 1: exogenous growing Y, tot energy proportional to Y
    #E = eps * Y
    E = 1 + eps * (Y - 1) # only for adimensional!!

    ### improve: energy demand is not all the same. energy for fossil-fuel cars, heavy industry, gas heating,... must be fossil. Electricity generation can easily be both. Converting fossil-locked energy demand to green energy demand requires converting the downstream infrastructure as well, which requires more investment (and more energy..). This could be represented through the "fossil_constraint" strategy.

    ## Satisfying energy demand through green and fossil energy production. 
    Eg, Ef, success = define_Eg(E, Kg, Kf, a, b, f_heavy, Y, rule = rule, verbose = verbose, success = success)

    if energy_crisis_mode and success == 2:
        E_missing = E - (Ef + Eg)
        Y_lost += E_missing/eps # This is the cumulative Y lost due to energy shortages
        print(f'Energy crisis mode: reducing Y due to energy constraints. E missing: {E_missing}, Y lost: {Y_lost}')

        E = E - E_missing
        Y = Y - Y_lost

    ### PUBLIC INVESTMENT
    ## Profit of energy production
    Cg = prod_cost(Eg, eta_g, h_g, etamax, scale_costs = scale_costs)
    Cf = prod_cost(Ef, eta_f, h_f, etamax, scale_costs = scale_costs)

    if public_inv:
        S = r_inv_state * Y
        ### improve: make mu depend on the ratio of elasticities (not ready! missing a dynamics for Y_g, Y_f)
        # er = el_ratio(mu_g, mu_f) # ratio of elasticities
        # mu_state = beta_fun(0., er, delta_sig = delta_sig_state, ftype = betafun_type) # using beta_fun with beta_0 = 0
        Sg = mu_state * S
        Sf = (1-mu_state) * S
    else:
        Sg = 0
        Sf = 0

    gamma_g_dyn, gamma_f_dyn = E_price(Eg, Ef, Kg, Kf, gamma_g, gamma_f, a, b, alpha_util = alpha_util, dynamic_price = dynamic_price, same_price = same_price, min_price = min_price)
    Pg = gamma_g_dyn * (Eg - Cg) + Sg # Sg should act on Cg and be limited to it? no, also investment in infrastructuree
    Pf = gamma_f_dyn * (Ef - Cf) + Sf
    ### PRIVATE INVESTMENT

    ## Investment in energy production
    pr = prof_ratio(Pg, Pf, Kg, Kf)
    beta = beta_fun(beta_0, pr, delta_sig = delta_sig, ftype = betafun_type)
    
    Ig = beta * r_inv * (Pg + Pf)
    If = (1-beta) * r_inv * (Pg + Pf)
    if verbose: print(('check: ' + 8*'{:10.2f}').format(beta, pr, Eg, Ef, Pg, Pf, Ig, If))

    ## for next step
    ## Capital/infrastructure
    if verbose and Ig < Kg*delta_g: print(f'Green infrastructure decreasing! {Ig} < {Kg*delta_g}')
    if verbose and If < Kf*delta_f: print(f'Fossil infrastructure decreasing! {If} < {Kf*delta_f}')

    # Kg = Ig + Sg + Kg * (1-delta_g) # if S goes to infrastructure directly, it competes with private investment instead of favoring it
    # Kf = If + Sf + Kf * (1-delta_f)
    
    # updating Y and K for next timestep
    
    Kg = Ig + Kg * (1-delta_g)
    Kf = If + Kf * (1-delta_f)

    Y = GDP(Y, growth = growth, deltaY = deltaY, gdp_type = gdp_type, gdp_scenario = gdp_scenario, year = year_scenario, Y_lost = Y_lost)

    Kg, Kf, Eg, Ef, beta, E, Y = check_bounds(Kg, Kf, Eg, Ef, beta, E, Y, raise_err = raise_bnd_err)

    return Y, Kg, Kf, E, Eg, Ef, Ig, If, Pg, Pf, Cg, Cf, Y_lost, success


def el_ratio(mu_g, mu_f):
    """
    The function computes the ratio of elasticities of green and fossil capital to the respective public energy investment. mu_g = dY/dS_g, mu_f = dY/dS_f

    for now very trivial: mu_g and mu_f are exogenous. in potential, could be less trivial if we had an internal dynamics for Y_g and Y_f
    """

    er = (mu_g - mu_f)/(mu_g + mu_f)

    return er


def define_Eg(E, Kg, Kf, a, b, f_heavy, Y, rule = 'maxgreen', verbose = True, success = 0):
    # Energy and infrastructure
    Eg_max = a * Kg # a = 1
    Ef_max = b * Kf # b time dependent, exog. should decrease to 0

    if Eg_max + Ef_max < E: 
        success = 2
        if verbose: print(f'Energy scarcity! {Eg_max} {Ef_max} {E}')

        Eg = Eg_max
        Ef = Ef_max
        
        return Eg, Ef, success

    if rule == 'maxgreen':
        Eg = Eg_max
        Ef = E-Eg
        if Eg > E:
            Eg = E
            Ef = 0.
    elif rule == 'proportional':
        Eg = Kg/(Kg+Kf) * E
        Ef = Kf/(Kg+Kf) * E
    elif rule == 'fair':
        if Ef_max >= E/2.:
            Ef = E/2.
        else:
            Ef = Ef_max
        Eg = E - Ef
    elif rule == 'whole_capacity': # This makes Y useless
        Eg = Eg_max
        Ef = Ef_max
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
    
    return Eg, Ef, success


def backward_step(Y, Kg, Kf, params = default_params, rule = 'maxgreen', betafun_type = 'cdf', verbose = False, raise_bnd_err = False):
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
    while cond and ii < max_iter:
        if verbose: print('ITeration:', ii)
        Eg, Ef = define_Eg(E, Kgit, Kfit, a, b, f_heavy, rule = rule)

        ## Profit of energy production of previous step
        Pg = gamma_g * (Eg - eta_g * Eg**h_g)
        Pf = gamma_f * (Ef - eta_f * Ef**h_f)
        if Pf < 0.: Pf = gamma_f * (1 - eta_f) * Ef # linearity for small Ef
        if Pg < 0.: Pg = gamma_g * (1 - eta_g) * Eg # linearity for small Eg

        ## Investment in energy production
        beta = beta_fun(beta_0, (Pg/Kg - Pf/Kf)/(Pg/Kg+Pf/Kf), delta_sig = delta_sig, ftype = betafun_type)
        #if verbose: print(beta, (Pg/Kg - Pf/Kf)/(Pf/Kf), Eg, Ef, Pg, Pf)
        
        Ig = beta * r_inv * (Pg + Pf)
        If = (1-beta) * r_inv * (Pg + Pf)
        #if verbose: print(Ig, If)

        Kgit_old = Kgit
        Kfit_old = Kfit

        Kgit = (Kg - Ig)/(1-delta_g)
        Kfit = (Kf - If)/(1-delta_f)

        Kgit, Kfit, Eg, Ef, beta, E, Y = check_bounds(Kgit, Kfit, Eg, Ef, beta, E, Y, raise_err = raise_bnd_err)

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

    return Y, Kg, Kf, E, Eg, Ef, Ig, If, Pg, Pf, success


def check_bounds(Kg, Kf, Eg, Ef, beta, E, Y, raise_err = False):
    # try:
    #     input_vec = np.array([Kg, Kf, Eg, Ef, beta, E, Y])
    # except ValueError as err:
    #     print(Kg, Kf, Eg, Ef, beta, E, Y)
    #     raise ValueError(f'Problem in check bounds: {err}')
    input_vec = np.array([Kg, Kf, Eg, Ef, beta, E, Y])

    nams = np.array('Kg, Kf, Eg, Ef, beta, E, Y'.split())
    mins = np.array([0, 0, 0, 0, 0, 0, 0])
    maxs = np.array([Kg, Kf, E, E, 1., E, Y])

    if np.all(input_vec >= mins) and np.all(input_vec <= maxs):
        pass
    elif np.any(input_vec < mins):
        if raise_err: 
            raise ValueError('Below threshold!', nams[np.where(input_vec < mins)])
        else:
            print('Resetting to min val: ', nams[np.where(input_vec < mins)])
            input_vec[np.where(input_vec < mins)] = mins[np.where(input_vec < mins)]
    elif np.any(input_vec > maxs):
        if raise_err:
            raise ValueError('Above threshold!', nams[np.where(input_vec > maxs)])
        else:
            print('Resetting to max val: ', nams[np.where(input_vec > maxs)])
            input_vec[np.where(input_vec > maxs)] = maxs[np.where(input_vec > maxs)]

    return list(input_vec)


def set_params(params, years, verbose = False):
    okpar = default_params.copy()
    scenario_pars = []

    for par in okpar:
        if type(params[par]) == float:
            if params[par] != default_params[par]:
                if verbose: print(f'Changing default for param {par}')
                okpar[par] = params[par]
        else:
            okpar[par] = params[par]

        if isinstance(params[par], xr.core.dataarray.DataArray) or isinstance(params[par], np.ndarray):
            scenario_pars.append(par)

        # if f'{par}_linear' in params:
        #     if verbose: print(f'Setting value of {par} with linear slope!')
        #     intercept, slope = params[f'{par}_linear']
        #     okpar[par] = intercept + slope*(years - years[0]) # scenario
    
    if len(scenario_pars) > 0:
        allow_param_scenario = scenario_pars
    else:
        allow_param_scenario = None
    
    return okpar, allow_param_scenario


def run_model(inicond = default_inicond, params = default_params, n_iter = 100, rule = 'maxgreen', betafun_type = 'cdf', verbose = True, run_backwards = False, raise_bnd_err = False, year_ini = None, extend_constant = False, deltaY = None, public_investment = False, mu_state_scenario = None, scale_costs = False, gdp_type = 'exponential', gdp_scenario = None, energy_crisis_mode = False, dynamic_price = False, same_price = True):
    """

    Runs the model. Returns list of lists of outputs: [Y, Kg, Kf, E, Eg, Ef]  (can be improved!)

    Rules are for energy partition when potential production exceeds demand (see forward_step function).

    allow_param_scenario removed. if parameters are arrays or dataarrays, this flag is automatically activated.

    """
    if year_ini is None:
        raise ValueError(f'{year_ini} not set!')
    
    if run_backwards:
        raise ValueError('Removed, if needed uncomment code below')

    Y = inicond['Y_ini']
    Kg = inicond['Kg_ini']
    Kf = inicond['Kf_ini']
    Y_lost = 0

    years = np.arange(year_ini, year_ini + n_iter)
    params_ok, allow_param_scenario = set_params(params, years)
    okpar = params.copy()

    if public_investment:
        if mu_state_scenario is None: raise ValueError("Missing scenario for green share of public energy investment (mu_state_scenario)")

    resu = []
    for i in range(n_iter):
        # Saving Y and K for output
        Y_this = Y
        Kg_this = Kg
        Kf_this = Kf
        
        if allow_param_scenario is not None:
            for par in allow_param_scenario:
                if verbose: 
                    print(f'using scenario for param {par}')
                    #print(okpar[par])
                if isinstance(params_ok[par], xr.core.dataarray.DataArray):
                    ymax = params_ok[par].year.max().values
                    yok = min(year_ini + i, ymax)
                    #print(yok, ymax)
                    okpar[par] = params_ok[par].sel(year = yok).values
                else:
                    #print('checkpar', i)
                    okpar[par] = params_ok[par][i]

        if public_investment:
            ymax = mu_state_scenario.year.max().values
            yok = min(year_ini + i, ymax)
            mu_state = mu_state_scenario.sel(year = yok).values
        else:
            mu_state = None
            
        Y, Kg, Kf, E, Eg, Ef, Ig, If, Pg, Pf, Cg, Cf, Y_lost, success = forward_step(Y, Kg, Kf, params = okpar, verbose = verbose, rule = rule, betafun_type = betafun_type, raise_bnd_err= raise_bnd_err, deltaY = deltaY, mu_state = mu_state, public_inv = public_investment, scale_costs = scale_costs, gdp_type = gdp_type, gdp_scenario = gdp_scenario, year_scenario = year_ini + i+1, energy_crisis_mode = energy_crisis_mode, Y_lost = Y_lost, dynamic_price = dynamic_price, same_price = same_price)

        # if run_backwards: # removed compatibility
        #     Y, Kg, Kf, E, Eg, Ef, Ig, If, Pg, Pf, Cg, Cf, success = backward_step(Y, Kg, Kf, params = okpar, verbose = verbose, rule = rule, betafun_type = betafun_type, raise_bnd_err=raise_bnd_err)

        resu.append([Y_this, Kg_this, Kf_this, E, Eg, Ef, Ig, If, Pg, Pf, Cg, Cf])
        if success == 0: 
            continue
        elif success == 1:
            if verbose: print(f'Transition completed at time: {i}!')
            break
        elif success == 2:
            if verbose: print(f'Energy scarcity at time: {i}!')
            if energy_crisis_mode:
                if verbose: print('Energy crises mode: going on with reduced Y due to energy scarcity')
            else:
                break
    
    if extend_constant:
        if len(resu) < n_iter:
            if verbose: print(f'Too short! extending up to {year_ini + n_iter}')
            resu = np.stack(resu)
            last_row = resu[-1, :]        
            repeated = np.repeat(last_row[np.newaxis, :], n_iter - resu.shape[0], axis = 0)
            resu = np.concatenate([resu, repeated], axis = 0)

    resu = rebuild_resu(resu, run_backwards = run_backwards)
    
    if success == 2:
        resu['success'] = False
    else:
        resu['success'] = True
    
    if not run_backwards:
        if success == 1: 
            resu['transition'] = True
            resu['year_zero'] = i
            resu['year_peak'] = np.argmax(resu['Ef'])

            for ye in range(resu['year_peak'], len(resu['Ef'])):
                if resu['Ef'][ye] <= resu['Ef'][resu['year_peak']]/2.: break
            resu['year_halved'] = ye
            if verbose: print('Peak fossil: {}'.format(resu['year_peak']))
            if verbose: print('Halved fossil: {}'.format(resu['year_halved']))
        else:
            resu['transition'] = False
            resu['year_zero'] = np.nan
            resu['year_peak'] = np.nan
            resu['year_halved'] = np.nan
    
    if year_ini is not None:
        resu = build_resu_ds(resu, year_ini = year_ini)

    return resu


def rebuild_resu(resu, run_backwards = False):
    if isinstance(resu, list):
        resu = np.stack(resu)
    Ys = resu[:, 0]
    Kgs = resu[:, 1]
    Kfs = resu[:, 2]
    E = resu[:, 3]
    Eg = resu[:, 4]
    Ef = resu[:, 5]
    Ig = resu[:, 6]
    If = resu[:, 7]
    Pg = resu[:, 8]
    Pf = resu[:, 9]
    Cg = resu[:, 10]
    Cf = resu[:, 11]
    if run_backwards:
        raise ValueError('not supported')
        # Ys = Ys[::-1]
        # Kgs = Kgs[::-1]
        # Kfs = Kfs[::-1]
        # E = E[::-1]
        # Eg = Eg[::-1]
        # Ef = Ef[::-1]
        # Ig = Ig[::-1]
        # If = If[::-1]
        # Pg = Pg[::-1]
        # Pf = Pf[::-1]

    ok_resu = dict()
    ok_resu['Y'] = Ys
    ok_resu['Kg'] = Kgs
    ok_resu['Kf'] = Kfs
    ok_resu['E'] = E
    ok_resu['Eg'] = Eg
    ok_resu['Ef'] = Ef
    ok_resu['Ig'] = Ig
    ok_resu['If'] = If
    ok_resu['I'] = If + Ig
    ok_resu['Pg'] = Pg
    ok_resu['Pf'] = Pf
    ok_resu['Ig_ratio'] = Ig/(Ig+If)
    ok_resu['Eg_ratio'] = Eg/E
    ok_resu['Cg'] = Cg
    ok_resu['Cf'] = Cf

    return ok_resu


def build_resu_ds(resu, year_ini):
    data_vars = {vnam : (['year'], resu[vnam]) for vnam in resu.keys() if isinstance(resu[vnam], np.ndarray)}
    scalars = {vnam : resu[vnam] for vnam in resu.keys() if vnam not in data_vars}
    for ke in scalars:
        if 'year' in ke: scalars[ke] += year_ini
    years = np.arange(year_ini, year_ini + len(resu['Y']))

    ds = xr.Dataset(data_vars = data_vars, coords={'year': years}, attrs = scalars)

    return ds


def cost_function(parset, parnames = ['beta_0', 'gamma_g', 'growth', 'delta_sig'], params = default_params.copy(), year_ini = 2015, inicond = inicond_2015, verbose = False, obs = None, obs_weights = None, public_investment = False, mu_state_scenario = None, same_costs = False, scale_costs = False, same_price = False, recalc_inicond = False, deltaY = None, gdp_type = 'exponential', gdp_scenario = None, dynamic_price = False, same_delta = False, param_bounds = None, break_on_scarcity = False, cost_low = 0.5):
    """
    Fit model to (year_ini - 2025) obs.obs
    """

    large = 100.

    if verbose:
        print(obs, deltaY, public_investment, mu_state_scenario)
        
    n_iter = 2025 - year_ini
    years = np.arange(year_ini, 2025)

    pardict = {par: val for par, val in zip(parnames, parset)}
    if verbose:
        print('---------------------')
        print(pardict)

    for par in pardict:
        if 'intercept' in par:
            short_nam = par[:par.rfind('_')]
            if f'{short_nam}_slope' in parnames:
                params[short_nam] = pardict[f'{short_nam}_intercept'] + pardict[f'{short_nam}_slope']*(years - year_ini) # scenario
            else:
                raise ValueError(f'{short_nam}_slope not in parnames!')
        elif 'slope' in par:
            short_nam = par[:par.rfind('_')]
            if f'{short_nam}_intercept' not in parnames:
                raise ValueError(f'{short_nam}_intercept not in parnames!')
        else:
            params[par] = pardict[par]        

    # for parval, pnam in zip(ok_parset, ok_names):
    #         params[pnam] = parval
    #inicond = define_K_ini(inicond, params, fcu = 0.8)

    if recalc_inicond:
        #print('Recalculating inicond based on new parameters...')
        inicond_ok = inicond_yr(year_ini, params, adimensional = True, fcu = fossil_capacity_util)
    else:
        inicond_ok = inicond
    
    if same_costs:
        params['eta_f'] = params['eta_g']

    if same_price:
        params['gamma_f'] = params['gamma_g']

    if same_delta:
        params['delta_f'] = params['delta_g']

    # if param_bounds is not None:
    #     for par in pardict:
    #         if par in param_bounds:
    #             if pardict[par] < param_bounds[par][0] or pardict[par] > param_bounds[par][1]:
    #                 print(f'Param {par} out of bounds')
    #                 return large

    resu = run_model(inicond = inicond_ok, params = params, n_iter = n_iter, year_ini = year_ini, verbose = verbose, rule = 'maxgreen', extend_constant = True, deltaY = deltaY, public_investment = public_investment, mu_state_scenario = mu_state_scenario, scale_costs = scale_costs, gdp_type=gdp_type, gdp_scenario=gdp_scenario, dynamic_price = dynamic_price, same_price = same_price)

    cost = costfun(resu, obs, weights = obs_weights)

    # if break_on_scarcity: raise ValueError('scarcity')
    
    if not resu.success:
        if verbose: print(f'Not successful, returning {large}')
        return large

    if verbose: print(f'Cost: {cost}')

    if cost < cost_low:
        print('Cost: ', cost, 'params: ', params)

    return cost


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
        # Ig = np.diff(resu['Kg'])
        # If = np.diff(resu['Kf'])
        Ig = resu['Ig']
        If = resu['If']

        plt.plot((Ig/(Ig+If))[:20], label = 'model', color = 'black')
        plt.plot(Ig_obs/(Ig_obs+If_obs), label = 'obs', color = 'orange')

        colors = get_colors_from_colormap(len(all_resu))

        for resu, col in zip(all_resu, colors):
            # Ig = np.diff(resu['Kg'])
            # If = np.diff(resu['Kf'])
            Ig = resu['Ig']
            If = resu['If']
            plt.plot((Ig/(Ig+If))[:20], color = col, ls = '--', lw = 1)

            # plt.annotate(f'({x_annotate}, {y_annotate:.2f})', xy=(x_annotate, y_annotate), xytext=(x_annotate + 1, y_annotate - 0.5), arrowprops=dict(facecolor='black', shrink=0.05))

        plt.xlabel('time')
        plt.ylabel('Green share of energy investment (beta)')
        plt.legend()

        fig2 = plt.figure()
        resu = nominal
        plt.plot((resu['Eg']/resu['E'])[:20], label = 'model', color = 'black')
        plt.plot(Eg_ratio.sel(year = slice(2015, 2024)).values, label = 'obs', color = 'orange')
        
        for resu, col in zip(all_resu, colors):
            plt.plot((resu['Eg']/resu['E'])[:20], color = col, ls = '--', lw = 1)

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


def costfun(resu, obs, weights = None, verbose = False):
    """
    Generic cost function for whatever is inside obs. Resu is a dataset and obs is a dict of dataarrays with 'year' axis.

    If given, weights should be a dictionary with weights for all variables in obs.
    """

    large = 100.
    if not resu.success:
        if verbose: print(f'Not successful, returning {large}')
        return large
    
    cost = []

    if verbose:
        print('Resu:')
        print(resu)

        print('Obs:')
        print(obs)

    for var in obs:
        wvar = 1
        if weights is not None:
            if var in weights:
                wvar = weights[var]
            else:
                wvar = 1.
        
        cc = wvar*((resu[var]-obs[var])**2).sum().values
        cost.append(cc)

    return np.sum(cost)


def costfun_1524(resu, year_ini = 2015, I_weight = 1., all_green = False):
    """
    Calcs cost function to observed data for 2015-2024.

    year_ini indicates first year of model sim
    I_weight is the weight to give to the "investment part" of the cost function relative to the energy share part
    """
    Ig = resu['Ig']
    If = resu['If']

    if isinstance(resu, xr.core.dataset.Dataset):
        sim_pr = (Ig/(Ig+If)).sel(year = slice(2015, 2024)).values
        sim_eg = (resu['Eg']/resu['E']).sel(year = slice(2015, 2024)).values
    else:
        ind_ini = 2015 - year_ini
        ind_fin = ind_ini + 9
        sim_pr = (Ig/(Ig+If))[ind_ini:ind_fin]
        sim_eg = (resu['Eg']/resu['E'])[ind_ini:ind_fin]

    if all_green:
        cost_I = 1.e4*np.sum((sim_pr - (Ig_obs_all/(Ig_obs_all+If_obs)))**2)
    else:
        cost_I = 1.e4*np.sum((sim_pr - (Ig_obs/(Ig_obs+If_obs)))**2)

    cost_Eg = np.sum((sim_eg - Eg_ratio.sel(year = slice(2015, 2024)).values)**2)

    return I_weight * cost_I + cost_Eg


def costfun_hist(resu, year_ini = 2000, I_weight = 1., all_green = False):
    """
    Calcs cost function to observed data, using both energy share (1965-2024) and investment share (2015-2024).

    year_ini indicates first year of model sim
    I_weight is the weight to give to the "investment part" of the cost function relative to the energy share part

    """
    ind_ini = 2015 - year_ini
    ind_fin = ind_ini + 9

    Ig = resu['Ig']
    If = resu['If']

    #print(ind_ini, ind_fin, len(Ig))

    if all_green:
        cost_I = 1.e4*np.sum(((Ig/(Ig+If))[ind_ini:ind_fin] - (Ig_obs_all/(Ig_obs_all+If_obs)))**2)
    else:
        cost_I = 1.e4*np.sum(((Ig/(Ig+If))[ind_ini:ind_fin] - (Ig_obs/(Ig_obs+If_obs)))**2)

    ind_ini = 1965 - year_ini
    if ind_ini < 0: ind_ini = 0
    ind_fin = ind_ini + len(Eg_ratio)
    if ind_fin > len(resu['Eg']): ind_fin = len(resu['Eg'])-1

    #print(ind_ini, ind_fin, len(Ig))

    ind_obs_ini = year_ini - 1965
    ind_obs_fin = ind_obs_ini + len(resu['Eg'])-1

    #print(ind_obs_ini, ind_obs_fin, len(Ig))

    cost_Eg = np.sum(((resu['Eg']/resu['E'])[ind_ini:ind_fin] - Eg_ratio[ind_obs_ini:ind_obs_fin])**2)

    return I_weight * cost_I + cost_Eg


def plot_resuvsobs_ds(resu, obs, year_ok = slice(2000, 2030), var_names = None, run_names = [], greystyle = False, colors = [], alphas = [], lw = 0.2):
    """
    Generic plot function for whatever is inside obs. Resu is a dataset and obs is a dict of dataarrays with 'year' axis.
    """
        
    figs = []
    for var in obs:
        fig = plt.figure()

        if isinstance(resu, xr.Dataset):
            resupl = resu[var].sel(year = year_ok).plot(label = 'model', color = 'orange')
        elif isinstance(resu, list):
            if run_names == []:
                run_names = [f'run {i}' for i in range(len(resu))]

            if colors == []: colors = ['grey']*len(resu)
            if alphas == []: alphas = [1]*len(resu)

            for res, nam, col, alp in zip(resu, run_names, colors, alphas):
                if not greystyle:
                    resupl = res[var].sel(year = year_ok).plot(label = nam, zorder = 0)
                else:
                    resupl = res[var].sel(year = year_ok).plot(color = col, lw = lw, zorder = 0, alpha = alp)

        obspl = obs[var].sel(year = year_ok).plot(label = 'obs', color = 'black', zorder = 0, lw = 0.5)
        if greystyle:
            obs[var].sel(year = year_ok).plot.scatter(color = 'black', s = 50, zorder = 2, alpha = 0.7)

        plt.xlabel('year')
        if var_names is not None:
            plt.ylabel(var_names[var])
        else:
            plt.ylabel(var)

        if not greystyle: plt.legend()
        figs.append(fig)

    return figs


def plot_resuvsobs(resu, year_ini = 2000, year_fin = 2100, maxlen = None, all_green = False, mod_col = 'orange', obs_col = 'black', obs_name = 'obs', mod_name = 'model'):#, ind_ini = 0, ind_fin = 20):
    """
    Plots outputs vs observed green investment and green energy share.
    """

    if maxlen is not None:
        year_ini = resu.year[0]
        year_fin = resu.year[0] + maxlen

    if not isinstance(resu, xr.core.dataset.Dataset):
        resu = build_resu_ds(resu, year_ini)

    fig = plt.figure()
    # Ig = np.diff(resu['Kg'])
    # If = np.diff(resu['Kf'])

    #totle = min(maxlen, len(Ig))
    #resu = resu.isel(year = slice(0, maxlen))
    resu = resu.sel(year = slice(year_ini, year_fin))

    Ig = resu['Ig']
    If = resu['If']

    resu['beta'] = resu.Ig/(resu.If + resu.Ig)

    resu.beta.plot(label = mod_name, color = mod_col)
    # plt.plot(np.arange(year_ini, year_ini + totle), (Ig/(Ig+If))[:totle], label = mod_name, color = mod_col)
    if all_green:
        print('Plotting original data of green investment from world bank')
        Ig_ratio_obs = Ig_obs_all/(Ig_obs_all+If_obs)
        Ig_ratio_obs.sel(year = slice(year_ini, year_fin)).plot(label = obs_name, color = obs_col)
        #plt.plot(np.arange(2015, 2024), Ig_obs_all/(Ig_obs_all+If_obs), label = obs_name, color = obs_col)
    else:
        print('Plotting only data regarding investment on green power production (only part of what world bank considers green investment)')
        Ig_ratio_obs = Ig_obs/(Ig_obs+If_obs)
        Ig_ratio_obs.sel(year = slice(year_ini, year_fin)).plot(label = obs_name, color = obs_col)
        #plt.plot(np.arange(2015, 2024), Ig_obs/(Ig_obs+If_obs), label = obs_name, color = obs_col)

    plt.xlabel('year')
    plt.ylabel(r'Green share of energy investment ($\beta$)')
    plt.legend()

    fig2 = plt.figure()
    resu['Eg_ratio'] = resu['Eg']/resu['E']
    Eg_ratio.sel(year = slice(year_ini, year_fin)).plot(label = obs_name, color = obs_col)
    resu['Eg_ratio'].plot(label = mod_name, color = mod_col)
    # plt.plot(np.arange(year_ini, year_ini + totle), (resu['Eg']/resu['E'])[:totle], label = mod_name, color = mod_col)
    # plt.plot(np.arange(year_ini, 2024), Eg_ratio_fe[-(2024-year_ini):], label = obs_name, color = obs_col)

    plt.xlabel('year')
    plt.ylabel('Share of renewable energy')
    plt.legend()

    return fig, fig2


def plot_hist(resu, year_ini = 1950, maxlen = 50):
    """
    Plots outputs vs observed green investment and green energy share.
    """

    fig = plt.figure()
    #Ig = np.diff(resu['Kg'])
    #If = np.diff(resu['Kf'])
    Ig = resu['Ig']
    If = resu['If']

    totle = min(maxlen, len(Ig))

    plt.plot(np.arange(year_ini, year_ini + totle), (Ig/(Ig+If))[:totle], label = 'model', color = 'black')
    plt.plot(np.arange(2015, 2024), Ig_obs/(Ig_obs+If_obs), label = 'obs', color = 'orange')

    plt.xlabel('time')
    plt.ylabel('Green share of energy investment (beta)')
    plt.legend()

    fig2 = plt.figure()
    plt.plot(np.arange(year_ini, year_ini + totle), (resu['Eg']/resu['E'])[:totle], label = 'model', color = 'black')
    plt.plot(np.arange(1965, 2024), Eg_ratio, label = 'obs', color = 'orange')

    plt.xlabel('time')
    plt.ylabel('Share of renewable energy')
    plt.legend()

    return fig, fig2


def plot_resu(resu, year_ini = None, title = None):
    if not isinstance(resu, xr.core.dataset.Dataset):
        if year_ini is not None:
            resu = build_resu_ds(resu, year_ini)
            xax = resu.year
        else:
            xax = np.arange(len(resu['E']))
    else:
        xax = resu.year

    fig, ax = plt.subplots()
    plt.plot(xax, resu['Kf'] + resu['Kg'], label = 'Total')
    plt.plot(xax, resu['Kf'], label = 'Fossil')
    plt.plot(xax, resu['Kg'], label = 'Green')
    if year_ini is not None:
        plt.xlabel('year')
    else:
        plt.xlabel('time')
    plt.ylabel('Energy infrastructure')
    plt.legend()
    if title is not None:
        plt.title(title)

    fig2, ax2 = plt.subplots()
    plt.plot(xax, resu['E'], label = 'Total')
    plt.plot(xax, resu['Ef'], label = 'Fossil')
    plt.plot(xax, resu['Eg'], label = 'Green')

    if not np.isnan(resu.year_peak):
        ax2.axvline(resu.year_peak, color = 'indianred', lw = 0.5, ls = ':')
    if not np.isnan(resu.year_halved):
        ax2.axvline(resu.year_halved, color = 'grey', lw = 0.5, ls = ':')
    if not np.isnan(resu.year_zero):
        ax2.axvline(resu.year_zero, color = 'forestgreen', lw = 0.5, ls = ':')

    if year_ini is not None:
        plt.xlabel('year')
    else:
        plt.xlabel('time')
    plt.ylabel('Energy production')
    plt.legend()
    
    if title is not None:
        plt.title(title)

    return fig, fig2


######## Tuning
import re
import ast

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

def parse_cost_and_params_line(line):
    # Extract the dictionary part
    match = re.search(r'params:\s*(\{.*\})', line)
    if not match:
        return None, None
    
    # Extract cost
    cost_match = re.search(r'Cost:\s*([\d.]+)', line)
    cost = float(cost_match.group(1)) if cost_match else None
    
    # Get the dictionary string
    dict_str = match.group(1)
    
    # Replace np.float64(...) with just the number
    dict_str = re.sub(r'np\.float64\(([^)]+)\)', r'\1', dict_str)
    
    # Now safely evaluate
    try:
        params_dict = ast.literal_eval(dict_str)
    except:
        # If that fails, use eval with numpy in namespace
        params_dict = eval(dict_str, {"np": np, "__builtins__": None})
    
    return cost, params_dict

# Usage with file
def parse_file_with_params(filename):
    costs = []
    param_sets = []
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('Cost:'):
                cost, params = parse_cost_and_params_line(line)
                if params:
                    costs.append(cost)
                    param_sets.append(params)
    
    return costs, param_sets


def parse_line(line):
    """Parse a line and extract cost and params dict"""
    # Extract cost
    cost_match = re.search(r'Cost:\s+([\d.e+-]+)', line)
    if not cost_match:
        return None, None
    cost = float(cost_match.group(1))
    
    # Extract params dict string
    params_match = re.search(r'params:\s+(\{.+\})', line)
    if not params_match:
        return None, None
    
    params_str = params_match.group(1)
    
    # Convert np.float64(...) to regular floats for eval
    params_str = re.sub(r'np\.float64\(([\d.e+-]+)\)', r'\1', params_str)
    
    # Safely evaluate the dict
    try:
        params = eval(params_str)
        return cost, params
    except:
        return None, None


def read_costs_from_log(filename):
    """
    Read txt file and extract costs and params, segmented by "AAAAAAA" lines
    
    Args:
        filename: path to txt file
    
    Returns:
        all_costs: list of cost lists (one per segment)
        all_params: list of param lists (one per segment)
    """
    all_costs = []
    all_params = []
    
    current_costs = []
    current_params = []
    
    with open(filename, 'r') as f:
        for line in f:
            # Check for segment separator
            if line.strip().startswith("AAAAAAA"):
                # Save current segment if not empty
                if current_costs:
                    all_costs.append(current_costs)
                    all_params.append(current_params)
                # Start new segment
                current_costs = []
                current_params = []
            else:
                # Parse line
                cost, param = parse_line(line)
                if cost is not None:
                    current_costs.append(cost)
                    current_params.append(param)
    
    # Add last segment if not empty
    if current_costs:
        all_costs.append(current_costs)
        all_params.append(current_params)
    
    return all_costs, all_params


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


def filter_costs(costs, params, percentile=10):
    """
    Filter costs and params by percentile threshold
    
    Args:
        costs: list of costs
        params: list of param dicts
        percentile: keep only entries with cost <= this percentile (default 10)
    
    Returns:
        costs_filtered: list of costs in the percentile
        params_filtered: list of param dicts in the percentile
    """
    # Calculate percentile threshold
    threshold = np.percentile(costs, percentile)
    
    # Filter by percentile
    costs_filtered = []
    params_filtered = []
    
    for cost, param in zip(costs, params):
        if cost <= threshold:
            costs_filtered.append(cost)
            params_filtered.append(param)
    
    return costs_filtered, params_filtered

def filter_by_bounds(costs, params, param_bounds):
    """
    Filter out entries where parameters are outside specified bounds
    
    Args:
        costs: list of costs
        params: list of param dicts
        param_bounds: dict with param names as keys and (min, max) tuples as values
                     Example: {'a': (0.5, 1.0), 'b': (0.0, 2.0)}
    
    Returns:
        costs_filtered: list of costs within bounds
        params_filtered: list of param dicts within bounds
    """
    
    costs_filtered = []
    params_filtered = []
    
    for cost, pardict in zip(costs, params):
        # Check if all params are within bounds
        within_bounds = True
        for par in pardict:
            if par in param_bounds:
                if pardict[par] < param_bounds[par][0] or pardict[par] > param_bounds[par][1]:
                    within_bounds = False
                    break
        
        if within_bounds:
            costs_filtered.append(cost)
            params_filtered.append(pardict)
    
    return costs_filtered, params_filtered


def plot_param_histograms(params):
    """Plot histogram for each parameter in a multi-panel figure"""
    
    # Get all parameter keys
    param_keys = sorted(set(k for p in params for k in p.keys()))
    
    # Calculate grid dimensions
    n_params = len(param_keys)
    n_cols = int(np.ceil(np.sqrt(n_params)))
    n_rows = int(np.ceil(n_params / n_cols))
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 2))
    axes = axes.flatten() if n_params > 1 else [axes]
    
    # Plot each parameter
    for idx, key in enumerate(param_keys):
        values = [p[key] for p in params if key in p]
        axes[idx].hist(values, bins=20, edgecolor='black')
        axes[idx].set_title(key)
        axes[idx].set_ylabel('Count')
    
    # Hide unused subplots
    for idx in range(n_params, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    return fig
