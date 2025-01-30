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

years   = [2024] 
months  = [3,4] # 
days    = np.arange(1,32)   # [14,15]   # 



fs = 20 # Hz, sampling frequency

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


metar =     pd.read_csv('Tonopah_METAR.csv',
                    index_col = 1,
                    header = 0,    
                    #skiprows = [0], 
                    engine = 'c',
                    on_bad_lines='warn', 
                    parse_dates=True
                        )   








#%% Overview

mast_vs_metar = 1
if mast_vs_metar == 1:
    
    all = metar.merge(inflow, left_index=True, right_index=True, how="inner")
    
    
    # Timeseries
    fig = plt.figure(figsize=(15,9))
    plt.suptitle(f"{inflow.index[0].date()} to {inflow.index[-1].date()}")
    plt.title('Wind dir ($^\circ$)')
    plt.plot(all.wdir_Top, all.drct,".", ms=3) 
    plt.xlabel("Inflow mast 11m")
    plt.ylabel("Metar")
    plt.grid()
    
    

    


    
    

                    
    







    







    
 

