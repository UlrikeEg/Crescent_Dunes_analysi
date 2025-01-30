import pandas as pd
import s3fs
import numpy as np
import pyarrow
from windrose import WindroseAxes
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.dates as mdates
import scipy as sp
import scipy.signal
import sys
import time
import glob
import pickle
import datetime
import matplotlib.cm as cm
from matplotlib import rcParams, rcParamsDefault


plt.rc('font', size=14) 



sys.path.append("C:/Users/uegerer/Desktop/NSO/NSO_data_processing")
sys.path.append("C:/Users/uegerer/Desktop/NSO/NSO_analysis")
sys.path.append("C:/Users/uegerer/Desktop/NSO/NSO_loads_processing")

from Functions_general import *
from Functions_loads import *

from wind_env import *






#%% Definitions and read data



# Define date to read data (can be multiple options with '*')
year = '*'
month = '*'
day = '*'


# Read the datasets for all dates

years   = [2024, 2025] 
months  = [3,4, 9,10,11,12,1] # 
days    = np.arange(1,32)   # [14,15]   # 



path = 'Y:\Wind-data/Restricted/Projects/NSO/CrescentDunes_processed_met_tower_data_preliminary/'  # 'CD_processed_data_preliminary/'    #   


inflow_files = []
mast_files = []
for year in years:
    year = str(year)

    for month in months:
        
        if year == '2024' and month<3: # no data before March 2024
            continue
        
        month = f'{month:02d}'
    
        for day in days:
            day = f'{day:02d}'
              
            print (year, month, day) 
            inflow_files = inflow_files + sorted(glob.glob(path +'Inflow_Mast_1min_' + year + '-' + month + '-' + day + '_' + '*.pkl'))   #
            mast_files = mast_files + sorted(glob.glob(path +'Wake_masts_1min_' + year + '-' + month + '-' + day + '_' + '*.pkl'))   #        



### Read data
inflow = pd.DataFrame()    
for datafile in inflow_files:
    inflow = pd.concat( [inflow, pd.read_pickle(datafile)]) 
inflow = inflow.sort_index()


masts = pd.DataFrame()
for datafile in mast_files:
    masts = pd.concat( [masts, pd.read_pickle(datafile)] )   
masts = masts.sort_index()




### Filter wind directions

# Use only wind direction when wind speed greater than defined wind speed limit
wind_dir_limit = 5 # m/s
for col_dir in [col for col in inflow.columns if 'wdir_' in col]:    # loop over every inflow mast height      
    col_spd = col_dir.replace("wdir", "wspd") # column name for wind speed
    inflow[col_dir] = inflow[col_dir].where(inflow[col_spd] > wind_dir_limit)
for col_dir in [col for col in masts.columns if 'wdir_' in col]:    # loop over every inflow mast height      
    col_spd = col_dir.replace("wdir", "wspd") # column name for wind speed
    masts[col_dir] = masts[col_dir].where(masts[col_spd] > wind_dir_limit)
    
    
    
    
    
    
# METAR


# metar =     pd.read_csv('Tonopah_METAR.csv',
#                     index_col = 1,
#                     header = 0,    
#                     #skiprows = [0], 
#                     engine = 'c',
#                     on_bad_lines='warn', 
#                     parse_dates=True
#                         )   




# Read Metar data from Tonopah station
def read_metar(station, start_year, start_month, start_day, end_year, end_month, end_day):
    
    """
    use: 
        
    met = read_metar(station='TPH', 
                       start_year=year, start_month=month, start_day=day, 
                       end_year=day_after.year, end_month=day_after.month, end_day=day_after.day)
    
    year, month, date can be either strings or integers
    """
        
    import requests
    from io import StringIO
    
    # API endpoint (https://mesonet.agron.iastate.edu/request/download.phtml?network=NV_ASOS)
    url = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
    params = {
        'data': 'tmpf,dwpf,relh,drct,sknt,p01i,alti,mslp,vsby,gust,peak_wind_gust,peak_wind_drct,peak_wind_time',
        'station': station,
        'tz': 'UTC',
        'year1': start_year,
        'month1': start_month,
        'day1': start_day,
        'year2': end_year,
        'month2': end_month,
        'day2': end_day
    }
    
    # Send a GET request to the API
    response = requests.get(url, params=params)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Load the response content into a DataFrame
        csv_data = StringIO(response.text)
        metar = pd.read_csv(csv_data, delimiter=',', index_col=1, parse_dates=True, na_values="M")
    else:
        metar = pd.DtaFrame()
        print(f"Failed to retrieve Metar data: {response.status_code}")
        
    return metar



metar = read_metar(station='TPH', 
                   start_year=2024, start_month=3, start_day=10, 
                   end_year=2025, end_month=1, end_day=10)


metar.drct = metar.drct.where(metar.sknt/1.94384 > wind_dir_limit)
metar["wspd"] =metar.sknt/1.94384

#%% Overview

mast_vs_metar = 0
if mast_vs_metar == 1:
    
    all = metar.merge(inflow, left_index=True, right_index=True, how="inner")
    
    
    # Correlation
    fig = plt.figure(figsize=(15,6))
    plt.suptitle(f"{inflow.index[0].date()} to {inflow.index[-1].date()}")
    
    ax = plt.subplot(1, 2,  1)
    
    plt.title('Wind speed (m/s)')
    R = all.wspd_Top.corr(all.wspd).round(2)
    plt.plot(all.wspd_Top, all.wspd,".", ms=3, label = "R=" +  str( R )) 
    plt.xlabel("Inflow mast 11m")
    plt.ylabel("Metar")
    plt.grid()
    plt.legend()
    
    ax = plt.subplot(1, 2, 2)
    
    plt.title('Wind dir ($^\circ$)')
    R = all.wdir_Top.corr(all.drct).round(2)
    plt.plot(all.wdir_Top, all.drct,".", ms=3, label = "R=" +  str( R )) 
    plt.xlabel("Inflow mast 11m")
    plt.ylabel("Metar")
    plt.grid()
    plt.legend()
    
    
    
    # Time series
    
    from brokenaxes import brokenaxes
    import matplotlib.gridspec as gridspec
        
    fig = plt.figure(figsize=(15,9))
    plt.suptitle(f"{inflow.index[0].date()} to {inflow.index[-1].date()}")
    
    gs = gridspec.GridSpec(1, 2, figure=fig)  # Define a grid for two subplots

    bax = brokenaxes(
        xlims=((all.index.min(), pd.Timestamp('2024-05-01') ), ( pd.Timestamp('2024-09-10'), all.index.max())),
        hspace=0.05,  # Adjust spacing between broken sections
        subplot_spec=gs[0]  # Link the brokenaxes to the specific subplot
        )
    
    bax.set_title('Wind direction')
    bax.plot(all.wdir_Top,".", ms=3, label = "Inflow mast 11m") 
    bax.plot(all.drct,".", ms=3, label = "Tonopah") 
    bax.set_ylabel("Wind dir ($^\circ$)")
    bax.grid()
    bax.legend()
    
    # Add the box/frame around the brokenaxes plot
    for spine in bax.big_ax.spines.values():
        spine.set_visible(True)
        
    for ax in bax.axs:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.tick_params(axis='x', rotation=45)  # Rotate the labels   
        
    gs = gridspec.GridSpec(1, 2, figure=fig)  # Define a grid for two subplots

    bax = brokenaxes(
        xlims=((all.index.min(), pd.Timestamp('2024-05-01') ), ( pd.Timestamp('2024-09-10'), all.index.max())),
        hspace=0.001,  # Adjust spacing between broken sections
        subplot_spec=gs[1]  # Link the brokenaxes to the specific subplot
        )
    
    bax.set_title('Wind speed')
    bax.plot(all.wspd_Top,".", ms=3, label = "Inflow mast 11m") 
    bax.plot(all.wspd,".", ms=3, label = "Tonopah") 
    bax.set_ylabel("Wind speed (m/s)")
    bax.grid()
    bax.legend()
    
    # Add the box/frame around the brokenaxes plot
    for spine in bax.big_ax.spines.values():
        spine.set_visible(True)
    
    
    for ax in bax.axs:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.tick_params(axis='x', rotation=45)  # Rotate the labels    
    


wind_rose = 1

if wind_rose ==1:
    
    
    metar = metar[metar.index.isin(inflow.index)]
    
    
    fig = plt.figure(figsize=(14,6))
    bins = np.arange(0, 15, 2)
    
    ax = plt.subplot(1, 2,  1, projection="windrose")
    
    ws = metar['wspd']
    wd = metar['drct']
    

    plt.title( "Metar 10m {} to {}".format(metar.index[0].date(), metar.index[-1].date() ))
    ax.bar(wd, ws, normed=True, opening=0.8, cmap=cm.Spectral, bins=bins,nsector=50)
    #ax.set_legend(title="Wind speed 10m (m/s)", loc = 3)

    ax = plt.subplot(1, 2,  2, projection="windrose")
    
    ws = inflow['wspd_Top']
    wd = inflow['wdir_Top']

    plt.title( "Inflow mast 11m {} to {}".format(inflow.index[0].date(), inflow.index[-1].date() ))
    ax.bar(wd, ws, normed=True, opening=0.8, cmap=cm.Spectral, bins=bins,nsector=50)
    ax.set_legend(title="Wind speed (m/s)",    
                  loc='center left', 
                  bbox_to_anchor=(1.08, 0.5),  # Place legend outside the plot
                  fontsize=15  )

    fig.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.1)





    


    
    

                    
    







    







    
 

