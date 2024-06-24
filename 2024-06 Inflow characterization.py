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


"""

Cup anemometer .................. 15m
Top temperature..................... 11m
Top sonic................................ 11m
Mid sonic................................ 5.5m
Low Sonic............................... 2.75m
Temperature & Hu5mity........ 2.65m
Pressure................................. 1.3m
"""








#%% Definitions and read data



# Define date to read data (can be multiple options with '*')
year = '*'
month = '*'
day = '*'


# Read the datasets for all dates

years   = [2024] 
months  = [3, 4, 5] # 
days    = np.arange(1,31)   # [14,15]   # 



fs = 1 # Hz, sampling frequency

path = '../CD_postprocessing/CD_processed_data_preliminary/'    #'Y:\Wind-data/Restricted/Projects/NSO/CrescentDunes_processed_met_tower_data_preliminary/'  #    


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
            inflow_files = inflow_files + sorted(glob.glob(path +'inflow_Mast_1min_' + year + '-' + month + '-' + day + '_' + '*.pkl'))   #
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




# Rename columns to replace "11m", "5m", and "3m" with respective heights
inflow.columns = inflow.columns.str.replace('Top', '11m')
inflow.columns = inflow.columns.str.replace('Mid', '5m')
inflow.columns = inflow.columns.str.replace('Low', '3m')



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
    inflow_west = inflow[(inflow.wdir_11m > 225) & (inflow.wdir_11m < 315)]
elif wind_dir == 'south':   
    inflow_south = inflow[(inflow.wdir_11m > 135) & (inflow.wdir_11m < 225)]
elif wind_dir == 'north':   
    inflow_north = inflow[(inflow.wdir_11m > 315) | (inflow.wdir_11m < 45)]
    
# masts_west = masts.loc[masts.index.intersection(inflow_west.index)]
# loads_west = loads.loc[loads.index.intersection(inflow_west.index)]









#%% Overview

Overview = 0
if Overview == 1:
    
    print("Percentage of western winds during loads period:")
    print("inflow: {}%".format(round(len(inflow_west)/len(inflow)*100,1)))
    #print("Masts: {}%".format(round(len(masts_west)/len(masts)*100,1)))
    
    print("Average wind speed at 11m\n for all winds: {} m/s \n western winds: {} m/s".format(inflow.wspd_11m.mean(), inflow_west.wspd_11m.mean()))
    
    print("Average wind speed at 5.5m\n for all winds: {} m/s \n western winds: {} m/s".format(inflow.wspd_5m.mean(), inflow_west.wspd_5m.mean()))
    

    
    

    
   #  # Timeseries of wind
   #  fig = plt.figure(figsize=(11,4))
    
   #  ax2 = plt.subplot(2, 1, 1)
   #  plt.ylabel('Wind speed (m s$^{-1}$)')
   #  for height_col in [col for col in inflow.columns if ('wspd_3m' in col) & ("_m" not in col) & ("_s" not in col)]:    # loop over every mast height   
   #      ax2.plot(inflow[height_col],'.', label = height_col, color='black', ms=1)
   #   #   ax2.plot(inflow_west[height_col],'.', label = height_col, color='blue', ms=1)
   #  # for height_col in [col for col in masts.columns if ('m1_wspd_3m' in col) & ("_m" not in col) & ("_s" not in col)]:    # loop over every mast height   
   #  #     ax2.plot(masts_west[height_col],'.', label = height_col, color='orange', ms=1)
   #  plt.legend(loc=1,  markerscale=4)
   #  plt.grid()
   #  ax2.set_zorder(100)
    
   #  ax1 = plt.subplot(2, 1, 2, sharex=ax2)  # 
   #  ax1.set_ylabel('Wind dir ($^\circ$)')
   #  for height_col in [col for col in inflow.columns if 'wdir_3m' in col]:    # loop over every inflow mast height   
   #      ax1.plot(inflow[height_col],'.', label = "", color='black', ms=1)
   # #     ax1.plot(inflow_west[height_col],'.', label = "", color='blue', ms=1)
   #  ax1.set_ylim(0, 360)    
   #  plt.grid()    
    
   #  #plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d %H:%M'))
   #  fig.autofmt_xdate()
   #  ax1.set_xlabel("Time (UTC)")
   #  plt.tight_layout()
   #  fig.autofmt_xdate()
   #  plt.subplots_adjust(hspace=0.1)
    
   #  #fig.savefig('wind_paper_plots/Overview_wind_time_series.png')



    plt.rcParams.update({'font.size': 15})
    mpl.rcParams['lines.markersize'] = 1
                    
    
    fig = plt.figure(figsize=(15,9))
    fig.subplots_adjust(right=0.9)
   # plt.suptitle("{}".format(inflow.index[0].date()))
    ax1 = plt.subplot(3, 1, 2)
    ax1.set_ylabel('Wind dir ($^\circ$)')
    for height_col in [col for col in inflow.columns if ('wdir' in col)  & ('_m' not in col) & ('_s' not in col)]:    # loop over every inflow mast height      
        ax1.plot(inflow[height_col],'.', label = height_col)
    # for height_col in [col for col in masts_1min_cut.columns if ('wdir' in col)  & ('_m' not in col) & ('_s' not in col)]:    # loop over every wake mast height   
    #     ax1.plot(masts_1min_cut[height_col],'.', label = height_col)
    ax1.legend(loc=1, markerscale = 15)
    plt.grid()
    
    ax2 = plt.subplot(3, 1, 1, sharex=ax1)
    plt.ylabel('Wind speed (m s$^{-1}$)')
    plt.plot(inflow.WS_15m,'.', label = '15m', color="grey")                   
    for height_col in [col for col in inflow.columns if ('wspd' in col)  & ('_m' not in col) & ('_s' not in col)]:    # loop over every mast height   
        ax2.plot(inflow[height_col],'.', label = height_col)
    # for height_col in [col for col in masts_1min_cut.columns if ('wspd' in col)  & ('_m' not in col) & ('_s' not in col)]:    # loop over every wake mast height   
    #     ax2.plot(masts_1min_cut[height_col],'.', label = height_col)
    plt.legend(loc=1, markerscale = 15)
    plt.grid()
    
    ax2 = plt.subplot(3, 1, 3, sharex=ax1)
    plt.ylabel('Turbulence intensity, TI')               
    for height_col in [col for col in inflow.columns if ('TI' in col)  & ('_m' not in col) & ('_w' not in col)]:    # loop over every mast height   
        ax2.plot(inflow[height_col],'.', label = height_col)
    # for height_col in [col for col in masts_1min_cut.columns if ('wspd' in col)  & ('_m' not in col) & ('_s' not in col)]:    # loop over every wake mast height   
    #     ax2.plot(masts_1min_cut[height_col],'.', label = height_col)
    plt.legend(loc=1, markerscale = 15)
    plt.grid()
    
    # ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    # plt.ylabel('Temperature ($^\circ$)', color="C0")
    # plt.plot(inflow.Temp,".",label = 'Temp 2m', color="C0") 
    # # try:
    # #     plt.plot(inflow.Temp_11m,'.',label = 'Temp 11m', color="lightblue")
    # # except:
    # #     pass
    # #ax3.legend(loc=1).set_zorder(100)
    # ax4 = ax3.twinx()    
    # plt.plot(inflow.RH,'.',label = 'RH', color="C1") 
    # plt.ylabel('RH (%)', color="C1")  
    # ax5 = ax3.twinx()
    # ax5.spines.right.set_position(("axes", 1.05))
    # plt.plot(inflow.p,'.', label = 'p', color="grey")  
    # plt.ylabel('p (hPa)', color="grey")
    # plt.grid()    
    

    fig.autofmt_xdate()
    plt.subplots_adjust(hspace=0.001)
    plt.tight_layout()
    

   

    #%% Histograms
    fig2 = plt.figure(figsize=(13,8))
    plt.suptitle("Histograms {} to {}".format(inflow.index[0].date(), inflow.index[-1].date() ))
    ax1 = plt.subplot(3, 1, 2)
    #ax1.set_title("Wind direction histogram {} to {}".format(inflow.index[0].date(), inflow.index[-1].date() ))
    for height_col in [col for col in inflow.columns if ('wdir' in col)  & ('_m' not in col) & ('_s' not in col)]:    # loop over every mast height   
        plt.hist(inflow[height_col].dropna(),label = height_col, alpha=0.5, bins=180, density=True) 
    # for height_col in [col for col in masts_cut.columns if 'wdir_' in col]:    # loop over every mast height   
    #     plt.hist(masts_cut[height_col].dropna(),label = height_col, alpha=0.5, bins=180, density=True) 
    ax1.legend(loc="upper right")
    plt.grid()
    plt.xlabel('Wind direction ($^\circ$)')
        
    ax2 = plt.subplot(3, 1, 1)
    #ax2.set_title("Wind speed histogram {} to {}".format(inflow.index[0].date(), inflow.index[-1].date() ))
    for height_col in [col for col in inflow.columns if ('wspd' in col)  & ('_m' not in col) & ('_s' not in col)]:    # loop over every mast height   
        plt.hist(inflow[height_col].dropna(),label = height_col, alpha=0.5, bins=180, density=True) 
    # for height_col in [col for col in masts_cut.columns if 'wspd_' in col]:    # loop over every mast height   
    #     plt.hist(masts_cut[height_col].dropna(),label = height_col, alpha=0.5, bins=100, density=True) 
    ax2.legend(loc="upper right")
    plt.grid()
    plt.xlabel('Wind speed (m s$^{-1}$)')
    ax1.set_ylabel('Probability (%)')
    plt.tight_layout()


    ax3 = plt.subplot(3, 1, 3)
    #ax2.set_title("Wind speed histogram {} to {}".format(inflow.index[0].date(), inflow.index[-1].date() ))
    for height_col in [col for col in inflow.columns if ('TI' in col)  & ('_m' not in col) & ('_s' not in col)& ('_w' not in col)]:    # loop over every mast height   
        plt.hist(inflow[height_col].dropna(),label = height_col, alpha=0.5, bins=180, density=True) 
    for height_col in [col for col in inflow.columns if ('TI_w' in col)  & ('_m' not in col) & ('_s' not in col)]:    # loop over every mast height   
        plt.hist(inflow[height_col].dropna(),label = height_col, alpha=0.5, bins=180, density=True) 
    # for height_col in [col for col in masts_cut.columns if 'wspd_' in col]:    # loop over every mast height   
    #     plt.hist(masts_cut[height_col].dropna(),label = height_col, alpha=0.5, bins=100, density=True) 
    plt.legend()
    plt.grid()
    plt.xlabel('Turbulence intensity (%)')
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5)



#%% Wind roses for all masts    


Winds_over_troughs = 0
if Winds_over_troughs == 1:
    

    inflow_west_with_masts =  inflow
        

    nrows, ncols = 2, 4
    fig = plt.figure(figsize=(13,8))
    bins = np.arange(0, 14, 2)
    
    fig.suptitle("Masts from {} to {}".format(inflow.index[0].date(), inflow.index[-1].date()))

    
    ax = fig.add_subplot(nrows, ncols, 1, projection="windrose")
    ws = inflow_west_with_masts.dropna(subset=['wspd_11m', 'wdir_11m']).wspd_11m
    wd = inflow_west_with_masts.dropna(subset=['wspd_11m', 'wdir_11m']).wdir_11m
    ax.set_title("Inflow mast, 11m ".format(len(ws)))  # \n {} data points
    ax.bar(wd, ws, normed=True, opening=0.8, cmap=cm.Spectral, bins=bins, nsector=60)
    
    ax = fig.add_subplot(nrows, ncols, 5, projection="windrose")
    ws = inflow_west_with_masts.dropna(subset=['wspd_3m', 'wdir_3m']).wspd_3m
    wd = inflow_west_with_masts.dropna(subset=['wspd_3m', 'wdir_3m']).wdir_3m
    ax.set_title("Inflow mast, 3.5m".format(len(ws)))
    ax.bar(wd, ws, normed=True, opening=0.8, cmap=cm.Spectral, bins=bins, nsector=60)
    ax.set_legend(title="Wind speed (m/s)", loc = "center left", bbox_to_anchor=[1.2, 0.5])   
      
    
    
    bins = np.arange(0, 0.6, 0.1)
    ax = fig.add_subplot(nrows, ncols, 3, projection="windrose")
    ws = inflow_west_with_masts.dropna(subset=['TI_11m', 'wdir_11m']).TI_11m
    wd = inflow_west_with_masts.dropna(subset=['TI_11m', 'wdir_11m']).wdir_11m
    ax.set_title("Inflow mast, 11m ".format(len(ws)))  # \n {} data points
    ax.bar(wd, ws, normed=True, opening=0.8, cmap=cm.viridis_r, bins=bins, nsector=60)
    
    ax = fig.add_subplot(nrows, ncols, 7, projection="windrose")
    ws = inflow_west_with_masts.dropna(subset=['TI_3m', 'wdir_3m']).TI_3m
    wd = inflow_west_with_masts.dropna(subset=['TI_3m', 'wdir_3m']).wdir_3m
    ax.set_title("Inflow mast, 3.5m".format(len(ws)))
    ax.bar(wd, ws, normed=True, opening=0.8, cmap=cm.viridis_r, bins=bins, nsector=60)
    ax.set_legend(title="TI", loc = "center left", bbox_to_anchor=[1.2, 0.5])   
             

    #fig.savefig('wind_paper_plots/Wind_over_troughs.png', dpi=300) 



#%% Quantify stability

stab = 1
if stab == 1:
    
 
    fig = plt.figure()
 
    ax1 = plt.subplot(1, 1, 1)
    ax1.set_xlabel('R f')
    ax1.set_ylabel('z/L')
    c = inflow.Temp_11m-inflow.Temp   #inflow.wdir_11m  #  
    sc = ax1.scatter(inflow.R_f, inflow.zL, c = c, alpha=0.9, s=5, cmap='twilight_shifted')
    plt.yscale("symlog", linthresh=1e-1)
    plt.xscale("symlog", linthresh=1e-1)
    plt.grid()
    cbar = plt.colorbar(sc)
    cbar.set_label('Temperature difference 11m - 2m (deg C)')
    plt.tight_layout()
    

    fig = plt.figure()
 
    ax1 = plt.subplot(1, 1, 1)
    ax1.set_xlabel('wind speed')
    ax1.set_ylabel('z/L')
    c = inflow.Temp_11m-inflow.Temp   #inflow.wdir_11m  #  
    sc = ax1.scatter(inflow.wspd_11m, inflow.R_f, c = c, alpha=0.9, s=5, cmap='twilight_shifted')
    plt.yscale("linear")
    plt.xscale("linear")
    plt.grid()
    cbar = plt.colorbar(sc)
    cbar.set_label('Temperature difference 11m - 2m (deg C)')
    plt.tight_layout()


#%% TI vs TKE

ti_vs_tke = 0
if ti_vs_tke == 1:
    
 
    fig = plt.figure(figsize=(15,9))
 
    ax1 = plt.subplot(1, 2, 1)
    ax1.set_xlabel('Wind speed (m/s)')
    ax1.set_ylabel('TI')
    for height_col in [col for col in inflow.columns if ('wspd' in col)  & ('_m' not in col) & ('_s' not in col)]:    # loop over every inflow mast height      
        ax1.plot(inflow[height_col], inflow[height_col.replace("wspd", "TI")], '.', label = height_col, alpha=0.5)
    ax1.legend(loc=1, markerscale = 15)
    plt.grid()
    
    ax1 = plt.subplot(1, 2, 2)
    ax1.set_xlabel('Wind speed (m/s)')
    ax1.set_ylabel('TKE (m$^2$ / s$^2$)')
    for height_col in [col for col in inflow.columns if ('wspd' in col)  & ('_m' not in col) & ('_s' not in col)]:    # loop over every inflow mast height      
        ax1.plot(inflow[height_col], inflow[height_col.replace("wspd", "TKE")], '.', label = height_col, alpha=0.5)
    ax1.legend(loc=1, markerscale = 15)
    plt.grid()
    
    
    
    
#%% Diurnal cycle
    
diurnal =0
if diurnal == 1:
    
    
    inflow['local_time'] = inflow.index.tz_localize('UTC').tz_convert('US/Pacific')
    
    mpl.rcParams['lines.markersize'] = 8
    plt.rc('font', size=15) 
 
            
    spring = range(80, 172)
    summer = range(172, 264)
    fall = range(264, 355)
    
    def season(x):
        if x in spring:
           return 'Spring'
        if x in summer:
           return 'Summer'
        if x in fall:
           return 'Fall'
        else :
           return 'Winter'
    def assign_color(x):
        if x in spring:
           return 'green'
        if x in summer:
           return 'red'
        if x in fall:
           return 'orange'
        else :
           return 'blue'
       
    inflow['season'] = inflow.index.dayofyear.map(season)
    inflow['color'] = inflow.index.dayofyear.map(assign_color)
    

    # inflow['color'] = "C0"
    

    fig = plt.figure(figsize=(16,8))   
    i=3
    #plt.suptitle("Diurnal cycle at NSO")
    ax1 = plt.subplot(i, 2, 3)
    ax1.set_ylabel('Temp ($^\circ$C)')
    
    # ax2 = plt.subplot(5, 2, 2)
    # ax2.set_ylabel('RH (%)')
    
    ax3 = plt.subplot(i, 2, 4)
    ax3.set_ylabel('Stability $R_f$')
    ax3.set_ylim(-0.7,0.2)
    
    ax4 = plt.subplot(i, 2, 6)
    ax4.set_ylabel('Heat flux (W/m$^{2}$)')
    
    ax5 = plt.subplot(i, 2, 1)
    ax5.set_ylabel('Wind speed (m/s)')
    
    ax6 = plt.subplot(i, 2, 2)
    ax6.set_ylabel('Wind dir ($^\circ$)')
    
    # ax7 = plt.subplot(i, 2, 6)
    # ax7.set_ylabel('length scale $w$ (m)')
    
    # ax8 = plt.subplot(i, 2, 7)
    # ax8.set_ylabel('length scale $U$ (m)')

    # ax9 = plt.subplot(5, 2, 9)
    # ax9.set_ylabel('TI')
 #   ax9.grid(True)
    ax4.grid(True)
    ax4.set_xlabel('hour of local time')
    
    ax10 = plt.subplot(i, 2, 5)
    ax10.set_ylabel('TKE (m$^{2}$/s$^{2}$)')
    ax10.grid(True)
    ax10.set_xlabel('hour of local time')

    for Time, period in inflow.groupby( inflow.local_time.dt.time): 
        

        
      #  for season, period2 in period.groupby( period.season):  
            period2 = period
          
            color = period2.iloc[0].color

            ind = period2.local_time.dt.time.iloc[0].hour + period2.local_time.dt.time.iloc[0].minute/60
        
            print (ind)        
        
            ax1.plot([ind] * len(period2), period2.Temp,".",alpha=0.1, ms=0.5, label = "", color=color)            
            ax2.plot([ind] * len(period2), period2.RH,".",alpha=0.1, ms=0.5, color=color)           
            ax3.plot([ind] * len(period2), period2.R_f,".",alpha=0.5, ms=1, color=color)          
            ax4.plot([ind] * len(period2), period2.H_S,".",alpha=0.5, ms=1, color=color)            
            ax5.plot([ind] * len(period2), period2.wspd_11m,".",alpha=0.1, ms=0.5, color=color)            
            ax6.plot([ind] * len(period2), np.degrees(np.arctan2( - period2.V_ax_11m , period2.U_ax_11m)) + 180,".", color=color,alpha=0.1, ms=0.5)          
            #ax7.plot(ind, period2.ls_w_11m,".",alpha=0.1, ms=0.5 color=color)            
            #ax8.plot(ind, period2.ls_U_11m,".",alpha=0.1, ms=0.5 color=color)            
            # ax9.plot([ind] * len(period2), period2.TI_11m,".",alpha=0.1, ms=0.5, color=color)           
            ax10.semilogy([ind] * len(period2), period2.TKE_11m,".",alpha=0.1, ms=0.5, color=color) 
            
            
            ax1.plot(ind, period2.Temp.median(),".", label = "", color=color)            
            ax2.plot(ind, period2.RH.median(),".", color=color)           
            ax3.plot(ind, period2.R_f.median(),".", color=color)
            # ax3.plot(ind, period2.Ri_b.median(),"+", ms=5, color=color)             
            ax4.plot(ind, period2.H_S.median(),".", color=color)            
            ax5.plot(ind, period2.wspd_11m.median(),".", color=color)            
            ax6.plot(ind, np.degrees(np.arctan2( - period2.V_ax_11m.median() , period2.U_ax_11m.median())) + 180,".", color=color)          
            #ax7.plot(ind, period2.ls_w_11m.median(),".", color=color)            
            #ax8.plot(ind, period2.ls_U_11m.median(),".", color=color)            
            #ax9.plot(ind, period2.TI_11m.median(),".", color=color)           
            ax10.semilogy(ind, period2.TKE_11m.median(),".", color=color) 
            
    ax5.plot(np.nan, np.nan,".", label='spring', color='green')
    ax5.plot(np.nan, np.nan,".", label='summer', color='red')
    ax5.plot(np.nan, np.nan,".", label='fall', color='orange')
    ax5.plot(np.nan, np.nan,".", label='winter', color='blue')
    #ax5.legend()
    
    for ax in [ax1,ax2,ax3,ax5,ax5,ax6]:
        ax.set_xticklabels([]) 
        ax.set_xlabel("") 
        ax.grid(True)

    # ax3.set_ylim(-1, 1)
    plt.tight_layout()
    # fig.savefig('C:/Users/uegerer/Desktop/NSO/Quicklooks/relations/Diurnal cycle simple.png', dpi=300)  
    
    
    
    



#%% inflow wind profile
inflow_wind_profile = 1
if inflow_wind_profile == 1:
    
        ymin = 0
        ymax = 16
        
        # example day
        ex = inflow["2024-03-14" : "2024-03-13"].median()

         
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
        winds =        ['WS_15m','wspd_11m', 'wspd_5m', 'wspd_3m' ]
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
        plt.plot( [ex.WS_15m,ex.wspd_11m, ex.wspd_5m, ex.wspd_3m ], [15,11,5.5,2.75])
        plt.grid()  
        ax0.xaxis.set_major_formatter(FormatStrFormatter('%g'))
        # plt.xlabel('Wind direction inflow (m/s)')
        #plt.xlim(0,13)
        plt.ylim(ymin, ymax)     
        plt.ylabel("Height (m)")
        
        
        ## Wind direction
        ax0 = plt.subplot(1,3, 2)
        ax0.set_xlabel("TI")       
        heights =      [11,5.5,2.75]   
        winds =        ['TI_11m', 'TI_5m', 'TI_3m' ]
        winds_mean = []
        for (height, wind) in zip(heights, winds):
            #print (height, wind)
            winds_mean = winds_mean + [inflow[wind].median()]
            box = plt.boxplot(inflow[wind].dropna(), positions=[height], showfliers=False, vert=False, widths = 1, patch_artist=True, showmeans=True, 
                    boxprops=dict(fill=None, color=c1),
                    capprops=dict(color=c1),
                    whiskerprops=dict(color=c1),
                    medianprops=dict(color=c1),
                    meanprops={'markerfacecolor': c1, 'markeredgecolor': c1}
                    ) 
        plt.plot( [ex.wdir_11m, ex.wdir_5m, ex.wdir_3m ], [11,5.5,2.75])
        plt.grid()  
        ax0.xaxis.set_major_formatter(FormatStrFormatter('%g'))
        # plt.xlabel('Wind direction inflow (m/s)')
        #plt.xlim(0,13)
        plt.ylim(ymin, ymax)    
        ax0.set_yticklabels([]) 
        

        
        
        ## TKE
        ax0 = plt.subplot(1,3, 3)
        ax0.set_xlabel("TKE (m$^2$/s$^2$)")   
        winds =        ['TKE_11m', 'TKE_5m', 'TKE_3m' ]
        for (height, wind) in zip(heights, winds):
            #print (height, wind)
            plt.boxplot(inflow[wind].dropna(), positions=[height], showfliers=False, vert=False, widths = 1, patch_artist=True, showmeans=True,
                    boxprops=dict(fill=None, color=c1),
                    capprops=dict(color=c1),
                    whiskerprops=dict(color=c1),
                    medianprops=dict(color=c1),
                    meanprops={'markerfacecolor': c1, 'markeredgecolor': c1})  
        
        plt.plot( [ex.TKE_11m, ex.TKE_5m, ex.TKE_3m ], [11,5.5,2.75] )
        plt.grid()  
        ax0.xaxis.set_major_formatter(FormatStrFormatter('%g'))
        plt.ylim(ymin, ymax) 
        plt.xlim(0,6.6)
        ax0.set_yticklabels([]) 
        ax0.legend()
        
        plt.suptitle("Average wind conditions {} to {}".format(inflow.index[0].date(), inflow.index[-1].date()))
        
        plt.tight_layout()
       # plt.subplots_adjust( wspace=0.1, hspace=0.4)

 
    
 




        
