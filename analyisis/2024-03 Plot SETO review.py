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
months  = [3] # 
days    = np.arange(1,31)   # [14,15]   # 



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

# cut inflow, so that only data with sonics is used
inflow = inflow[:inflow.wspd_Low.dropna().index[-1]]

masts = pd.DataFrame()
for datafile in mast_files:
    masts = pd.concat( [masts, pd.read_pickle(datafile)] )   
masts = masts.sort_index()


# inflow.to_parquet("1min_data/inflow.parquet")
# masts.to_parquet("1min_data/masts.parquet")
# loads.to_parquet("1min_data/loads.parquet")


# inflow = pd.read_parquet("1min_data/inflow.parquet")
# masts = pd.read_parquet("1min_data/masts.parquet")
# loads = pd.read_parquet("1min_data/loads.parquet")


# inflow = inflow[loads.index[0]:loads.index[-1]]
# masts  = masts[loads.index[0]:loads.index[-1]]

inflow['rho']    = rho(inflow.p, inflow.RH, inflow.Temp).interpolate(limit_direction='both')
#inflow['u_star'] = (abs(inflow.Tau.interpolate()) / inflow.rho)**0.5 


### Filter wind directions

# Use only wind direction when wind speed greater than defined wind speed limit
wind_dir_limit = 0.5 # m/s
for col_dir in [col for col in inflow.columns if 'wdir_' in col]:    # loop over every inflow mast height      
    col_spd = col_dir.replace("wdir", "wspd") # column name for wind speed
    inflow[col_dir] = inflow[col_dir].where(inflow[col_spd] > wind_dir_limit)
for col_dir in [col for col in masts.columns if 'wdir_' in col]:    # loop over every inflow mast height      
    col_spd = col_dir.replace("wdir", "wspd") # column name for wind speed
    masts[col_dir] = masts[col_dir].where(masts[col_spd] > wind_dir_limit)
    
    
# Filter western winds
wind_dir = 'west'
if wind_dir == 'west':   
    inflow_west = inflow[(inflow.wdir_Top > 225) & (inflow.wdir_Top < 315)]
elif wind_dir == 'south':   
    inflow_south = inflow[(inflow.wdir_Top > 135) & (inflow.wdir_Top < 225)]
elif wind_dir == 'north':   
    inflow_north = inflow[(inflow.wdir_Top > 315) | (inflow.wdir_Top < 45)]
    
# masts_west = masts.loc[masts.index.intersection(inflow_west.index)]
# loads_west = loads.loc[loads.index.intersection(inflow_west.index)]











#%% Overview

Overview = 0
if Overview == 1:
    
    print("Percentage of western winds during loads period:")
    print("Inflow: {}%".format(round(len(inflow_west)/len(inflow)*100,1)))
    print("Masts: {}%".format(round(len(masts_west)/len(masts)*100,1)))
    
    print("Average wind speed at 7m\n for all winds: {} m/s \n western winds: {} m/s".format(inflow.wspd_Top.mean(), inflow_west.wspd_Top.mean()))
    
    print("Average wind speed at 3.4m\n for all winds: {} m/s \n western winds: {} m/s".format(inflow.wspd_Low.mean(), inflow_west.wspd_Low.mean()))
    
    inflow.wspd_Top_max.mean()
    inflow_west.wspd_Top_max.mean()  
    
    inflow.wspd_Low_max.max()
    inflow.wspd_Low.max()
    
    
    inflow.TI_U_Low.quantile(0.9)
    
    
    # Timeseries of wind
    fig = plt.figure(figsize=(11,4))
    
    ax2 = plt.subplot(2, 1, 1)
    plt.ylabel('Wind speed (m s$^{-1}$)')
    for height_col in [col for col in inflow.columns if ('wspd_Low' in col) & ("_m" not in col) & ("_s" not in col)]:    # loop over every mast height   
        ax2.plot(inflow[height_col],'.', label = height_col, color='black', ms=1)
     #   ax2.plot(inflow_west[height_col],'.', label = height_col, color='blue', ms=1)
    # for height_col in [col for col in masts.columns if ('m1_wspd_Low' in col) & ("_m" not in col) & ("_s" not in col)]:    # loop over every mast height   
    #     ax2.plot(masts_west[height_col],'.', label = height_col, color='orange', ms=1)
    plt.legend(loc=1,  markerscale=4)
    plt.grid()
    ax2.set_xticklabels([]) 
    ax2.set_zorder(100)
    
    ax1 = plt.subplot(2, 1, 2, sharex=ax2)  # 
    ax1.set_ylabel('Wind dir ($^\circ$)')
    for height_col in [col for col in inflow.columns if 'wdir_Low' in col]:    # loop over every inflow mast height   
        ax1.plot(inflow[height_col],'.', label = "", color='black', ms=1)
   #     ax1.plot(inflow_west[height_col],'.', label = "", color='blue', ms=1)
    ax1.set_ylim(0, 360)    
    plt.grid()    
    
    #plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d %H:%M'))
    fig.autofmt_xdate()
    ax1.set_xlabel("Time (UTC)")
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1)
    
    #fig.savefig('wind_paper_plots/Overview_wind_time_series.png')




                    
    







    









#%% Inflow wind profile
inflow_wind_profile = 1
if inflow_wind_profile == 1:
    
        ymin = 0
        ymax = 16
        
        # example day
        ex = inflow["2024-03-14" : "2024-03-14"].median()

         
        c11 = "red"
        c12 = "blue"
        c13 = "green"
        
        c1 = 'black'
    
        fig = plt.subplots( figsize=(12.0, 7.0) ) 
        plt.axis('off')
        
        ## Wind speed
        ax0 = plt.subplot(1,3, 1)
        ax0.set_xlabel("Wind speed (m/s)")
        heights =      [15,11,5.5,2.75]   
        winds =        ['WS_15m','wspd_Top', 'wspd_Mid', 'wspd_Low' ]
        winds_mean = []
        for (height, wind) in zip(heights, winds):
            #print (height, wind)
            winds_mean = winds_mean + [inflow[wind].mean()]
            box = plt.boxplot(inflow[wind].dropna(), positions=[height], showfliers=False, vert=False, widths = 1, patch_artist=True, showmeans=True, 
                    boxprops=dict(fill=None, color=c1),
                    capprops=dict(color=c1),
                    whiskerprops=dict(color=c1),
                    medianprops=dict(color=c1),
                    meanprops={'markerfacecolor': c1, 'markeredgecolor': c1}) 
        plt.plot( [ex.WS_15m,ex.wspd_Top, ex.wspd_Mid, ex.wspd_Low ], [15,11,5.5,2.75])
        plt.grid()  
        ax0.xaxis.set_major_formatter(FormatStrFormatter('%g'))
        # plt.xlabel('Wind direction inflow (m/s)')
        #plt.xlim(0,13)
        plt.ylim(ymin, ymax)     
        plt.ylabel("Height (m)")
        
        
        ## Wind direction
        ax0 = plt.subplot(1,3, 2)
        ax0.set_xlabel("Wind dir ($^\circ$)")       
        heights =      [11,5.5,2.75]   
        winds =        ['wdir_Top', 'wdir_Mid', 'wdir_Low' ]
        winds_mean = []
        for (height, wind) in zip(heights, winds):
            #print (height, wind)
            winds_mean = winds_mean + [inflow[wind].median()]
            box = plt.boxplot(inflow[wind].dropna(), positions=[height], showfliers=False, vert=False, widths = 1, patch_artist=True, showmeans=False, 
                    boxprops=dict(fill=None, color=c1),
                    capprops=dict(color=c1),
                    whiskerprops=dict(color=c1),
                    medianprops=dict(color=c1),
                    #meanprops={'markerfacecolor': c1, 'markeredgecolor': c1}
                    ) 
        plt.plot( [ex.wdir_Top, ex.wdir_Mid, ex.wdir_Low ], [11,5.5,2.75])
        plt.grid()  
        ax0.xaxis.set_major_formatter(FormatStrFormatter('%g'))
        # plt.xlabel('Wind direction inflow (m/s)')
        #plt.xlim(0,13)
        plt.ylim(ymin, ymax)    
        ax0.set_yticklabels([]) 
        

        
        
        ## TKE
        ax0 = plt.subplot(1,3, 3)
        ax0.set_xlabel("TKE (m$^2$/s$^2$)")   
        winds =        ['TKE_Top', 'TKE_Mid', 'TKE_Low' ]
        for (height, wind) in zip(heights, winds):
            #print (height, wind)
            plt.boxplot(inflow[wind].dropna(), positions=[height], showfliers=False, vert=False, widths = 1, patch_artist=True, showmeans=True,
                    boxprops=dict(fill=None, color=c1),
                    capprops=dict(color=c1),
                    whiskerprops=dict(color=c1),
                    medianprops=dict(color=c1),
                    meanprops={'markerfacecolor': c1, 'markeredgecolor': c1})  
        
        plt.plot( [ex.TKE_Top, ex.TKE_Mid, ex.TKE_Low ], [11,5.5,2.75], label = ex.orig_timestamp.date() )
        plt.grid()  
        ax0.xaxis.set_major_formatter(FormatStrFormatter('%g'))
        plt.ylim(ymin, ymax) 
        plt.xlim(0,6.6)
        ax0.set_yticklabels([]) 
        ax0.legend()
        
        plt.suptitle("Average wind conditions {} to {}".format(inflow.index[0].date(), inflow.index[-1].date()))
        
        plt.tight_layout()
       # plt.subplots_adjust( wspace=0.1, hspace=0.4)

 
    
 



#%% Wind roses for all masts    


Winds_over_troughs = 0
if Winds_over_troughs == 1:
    

    inflow_west_with_masts =  inflow#_west.loc[inflow_west.index.intersection(masts_west.index)]
    
    
    
    fig = plt.figure(figsize=(13,8))
    bins = np.arange(0, 14, 2)
    
    fig.suptitle("Inflow mast, 11m, at CD from {} to {}".format(inflow.index[0].date(), inflow.index[-1].date()))
    ax = fig.add_subplot(1, 1, 1, projection="windrose")
    # ws = inflow_west_with_masts.dropna(subset=['wspd_Top', 'wdir_Top']).wspd_Top
    # wd = inflow_west_with_masts.dropna(subset=['wspd_Top', 'wdir_Top']).wdir_Top
    ws = metar.dropna(subset=['sknt', 'drct']).sknt/1.94384
    wd = metar.dropna(subset=['sknt', 'drct']).drct
    ax.bar(wd, ws, normed=True, opening=0.8, cmap=cm.Spectral, bins=bins, nsector=60)
    ax.set_legend(title="Wind speed (m/s)", loc = "lower right")   
    
        

    nrows, ncols = 2, 4
    fig = plt.figure(figsize=(13,8))
    bins = np.arange(0, 14, 2)
    
    fig.suptitle("Masts from {} to {}, only ".format(inflow.index[0].date(), inflow.index[-1].date())  + wind_dir + " winds")
    fig.suptitle("Wind flow modification over parabolic troughs at western winds")
    
    ax = fig.add_subplot(nrows, ncols, 1, projection="windrose")
    ws = inflow_west_with_masts.dropna(subset=['wspd_Top', 'wdir_Top']).wspd_Top
    wd = inflow_west_with_masts.dropna(subset=['wspd_Top', 'wdir_Top']).wdir_Top
    ax.set_title("Inflow mast, 7m ".format(len(ws)))  # \n {} data points
    ax.bar(wd, ws, normed=True, opening=0.8, cmap=cm.Spectral, bins=bins, nsector=60)
    
    ax = fig.add_subplot(nrows, ncols, 5, projection="windrose")
    ws = inflow_west_with_masts.dropna(subset=['wspd_Low', 'wdir_Low']).wspd_Low
    wd = inflow_west_with_masts.dropna(subset=['wspd_Low', 'wdir_Low']).wdir_Low
    ax.set_title("Inflow mast, 3.5m".format(len(ws)))
    ax.bar(wd, ws, normed=True, opening=0.8, cmap=cm.Spectral, bins=bins, nsector=60)
    ax.set_legend(title="Wind speed (m/s)", loc = "lower right")   
        
    ax = fig.add_subplot(nrows, ncols, 2, projection="windrose")
    ws = masts_west.dropna(subset=['m1_wspd_Top', 'm1_wdir_Top']).m1_wspd_Top
    wd = masts_west.dropna(subset=['m1_wspd_Top', 'm1_wdir_Top']).m1_wdir_Top
    ax.set_title("Wake mast 1, 7m".format(len(ws)))
    ax.bar(wd, ws, normed=True, opening=0.8, cmap=cm.Spectral, bins=bins, nsector=60)   
    
    ax = fig.add_subplot(nrows, ncols, 6, projection="windrose")
    ws = pd.concat( [masts_west.dropna(subset=['m1_wspd_4m', 'm1_wdir_4m']).m1_wspd_4m, masts_west.dropna(subset=['m1_wspd_Low', 'm1_wdir_Low']).m1_wspd_Low])
    wd = pd.concat( [masts_west.dropna(subset=['m1_wspd_4m', 'm1_wdir_4m']).m1_wdir_4m, masts_west.dropna(subset=['m1_wspd_Low', 'm1_wdir_Low']).m1_wdir_Low])
    ax.set_title("Wake mast 1, 3.5m/4m".format(len(ws)))
    ax.bar(wd, ws, normed=True, opening=0.8, cmap=cm.Spectral, bins=bins, nsector=60)
    
    ax = fig.add_subplot(nrows, ncols, 3, projection="windrose")
    ws = masts_west.dropna(subset=['m2_wspd_Top', 'm2_wdir_Top']).m2_wspd_Top
    wd = masts_west.dropna(subset=['m2_wspd_Top', 'm2_wdir_Top']).m2_wdir_Top
    ax.set_title("Wake mast 2, 7m".format(len(ws)))
    if len (ws)>0:
        ax.bar(wd, ws, normed=True, opening=0.8, cmap=cm.Spectral, bins=bins, nsector=60)   
    
    ax = fig.add_subplot(nrows, ncols, 7, projection="windrose")
    ws = pd.concat( [masts_west.dropna(subset=['m2_wspd_4m', 'm2_wdir_4m']).m2_wspd_4m, masts_west.dropna(subset=['m2_wspd_Low', 'm2_wdir_Low']).m2_wspd_Low])
    wd = pd.concat( [masts_west.dropna(subset=['m2_wspd_4m', 'm2_wdir_4m']).m2_wdir_4m, masts_west.dropna(subset=['m2_wspd_Low', 'm2_wdir_Low']).m2_wdir_Low])
    ax.set_title("Wake mast 2, 3.5m/4m".format(len(ws)))
    if len (ws)>0:
        ax.bar(wd, ws, normed=True, opening=0.8, cmap=cm.Spectral, bins=bins, nsector=60)
    
    ax = fig.add_subplot(nrows, ncols, 4, projection="windrose")
    ws = masts_west.dropna(subset=['m3_wspd_Top', 'm3_wdir_Top']).m3_wspd_Top
    wd = masts_west.dropna(subset=['m3_wspd_Top', 'm3_wdir_Top']).m3_wdir_Top
    ax.set_title("Wake mast 3, 7m".format(len(ws)))
    if len (ws)>0:
        ax.bar(wd, ws, normed=True, opening=0.8, cmap=cm.Spectral, bins=bins, nsector=60)   
    
    ax = fig.add_subplot(nrows, ncols, 8, projection="windrose")
    ws = pd.concat( [masts_west.dropna(subset=['m3_wspd_4m', 'm3_wdir_4m']).m3_wspd_4m, masts_west.dropna(subset=['m3_wspd_Low', 'm3_wdir_Low']).m3_wspd_Low])
    wd = pd.concat( [masts_west.dropna(subset=['m3_wspd_4m', 'm3_wdir_4m']).m3_wdir_4m, masts_west.dropna(subset=['m3_wspd_Low', 'm3_wdir_Low']).m3_wdir_Low])
    ax.set_title("Wake mast 3, 3.5m/4m".format(len(ws)))
    if len (ws)>0:
        ax.bar(wd, ws, normed=True, opening=0.8, cmap=cm.Spectral, bins=bins, nsector=60)
             
    plt.tight_layout()
    
    #fig.savefig('wind_paper_plots/Wind_over_troughs.png', dpi=300) 
        
